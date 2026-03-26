"""
RockFrag Core - Motor de segmentación con SAM
Corregido para Streamlit Cloud:
  - Modelo SAM descargado en /tmp (evita errores de permisos)
  - Sin forzar CUDA (Streamlit Cloud no tiene GPU)
  - Cache del modelo con variable de módulo
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass, field
from typing import Optional
from ultralytics import SAM
import torch

# Forzar directorio de descarga de Ultralytics a /tmp
os.environ["YOLO_CONFIG_DIR"] = "/tmp/.ultralytics"

_SAM_INSTANCE = None   # Cache global para no recargar en cada análisis


@dataclass
class Fragment:
    id: int
    area_px: float
    area_cm2: float
    diameter_cm: float
    perimeter_px: float
    contour: np.ndarray
    bbox: tuple
    circularity: float


@dataclass
class AnalysisResult:
    image_path: str
    scale_px_per_cm: float
    fragments: list = field(default_factory=list)
    total_fragments: int = 0
    p20: float = 0.0
    p50: float = 0.0
    p80: float = 0.0
    mean_diameter: float = 0.0
    max_diameter: float = 0.0
    min_diameter: float = 0.0


class RockFragAnalyzer:
    def __init__(
        self,
        scale_reference_cm: float = 30.0,
        min_fragment_area_px: int = 200,
        max_fragment_ratio: float = 0.8,
        sam_model_path: str = "sam2_t.pt",
    ):
        self.scale_reference_cm = scale_reference_cm
        self.min_fragment_area_px = min_fragment_area_px
        self.max_fragment_ratio = max_fragment_ratio
        self.sam_model_path = sam_model_path

    def _load_sam(self):
        """Carga SAM una sola vez usando cache global de módulo."""
        global _SAM_INSTANCE
        if _SAM_INSTANCE is None:
            # Ultralytics descarga el modelo automáticamente a /tmp si no existe
            _SAM_INSTANCE = SAM(self.sam_model_path)
            # NO forzar CUDA — Streamlit Cloud corre en CPU
            # if torch.cuda.is_available():
            #     _SAM_INSTANCE.to('cuda')
        return _SAM_INSTANCE

    def detect_scale_bar(self, img: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > w * h * 0.05:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            if 3 < aspect < 20:
                candidates.append((cw, cnt))
        if candidates:
            bar_width_px = max(candidates, key=lambda c: c[0])[0]
            return bar_width_px / self.scale_reference_cm
        return None

    def analyze(
        self,
        image_path: str,
        scale_px_per_cm: Optional[float] = None,
    ) -> AnalysisResult:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {image_path}")

        h, w = img.shape[:2]

        if scale_px_per_cm is None:
            scale_px_per_cm = self.detect_scale_bar(img)
            if scale_px_per_cm is None:
                scale_px_per_cm = (w * 0.10) / self.scale_reference_cm

        sam = self._load_sam()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = sam(img_rgb)

        masks = []
        if results[0].masks is not None:
            masks_tensor = results[0].masks.data.cpu().numpy()
            for mask in masks_tensor:
                area = float(np.sum(mask))
                if area >= self.min_fragment_area_px:
                    masks.append(mask.astype(bool))

        if not masks:
            raise ValueError(
                "SAM no detectó fragmentos. Intenta reducir el 'Tamaño mínimo de fragmento' "
                "o verifica que la imagen contenga material de roca visible."
            )

        max_area_px = h * w * self.max_fragment_ratio
        fragments = []
        for i, mask in enumerate(masks):
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_fragment_area_px or area_px > max_area_px:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0
            area_cm2 = area_px / (scale_px_per_cm ** 2)
            diameter_cm = 2 * np.sqrt(area_cm2 / np.pi)
            bbox = cv2.boundingRect(cnt)
            fragments.append(Fragment(
                id=i,
                area_px=area_px,
                area_cm2=round(area_cm2, 2),
                diameter_cm=round(diameter_cm, 2),
                perimeter_px=round(perimeter, 1),
                contour=cnt,
                bbox=bbox,
                circularity=round(circularity, 3),
            ))

        if not fragments:
            raise ValueError(
                "Después de filtrar por tamaño no quedaron fragmentos. "
                "Ajusta los parámetros de tamaño mínimo/máximo."
            )

        fragments_sorted = sorted(fragments, key=lambda f: f.diameter_cm)
        diameters = [f.diameter_cm for f in fragments_sorted]

        return AnalysisResult(
            image_path=str(image_path),
            scale_px_per_cm=scale_px_per_cm,
            fragments=fragments_sorted,
            total_fragments=len(fragments_sorted),
            p20=float(np.percentile(diameters, 20)),
            p50=float(np.percentile(diameters, 50)),
            p80=float(np.percentile(diameters, 80)),
            mean_diameter=float(np.mean(diameters)),
            max_diameter=float(max(diameters)),
            min_diameter=float(min(diameters)),
        )


class RockFragVisualizer:
    @staticmethod
    def draw_segmentation(img: np.ndarray, result: AnalysisResult) -> np.ndarray:
        output = img.copy()
        n = len(result.fragments)
        for i, frag in enumerate(result.fragments):
            ratio = i / max(n - 1, 1)
            b = int(255 * (1 - ratio))
            r = int(255 * ratio)
            g = int(255 * (1 - abs(2 * ratio - 1)))
            color = (b, g, r)
            cv2.drawContours(output, [frag.contour], -1, color, 2)
            x, y, cw, ch = frag.bbox
            cx, cy = x + cw // 2, y + ch // 2
            label = f"{frag.diameter_cm:.1f}cm"
            cv2.putText(output, label, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return output

    @staticmethod
    def plot_grading_curve(result: AnalysisResult) -> bytes:
        diameters = sorted([f.diameter_cm for f in result.fragments])
        n = len(diameters)
        cumulative_pct = [(i + 1) / n * 100 for i in range(n)]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.plot(diameters, cumulative_pct, color='#00d4ff', linewidth=2.5, label='Curva granulométrica')
        ax.fill_between(diameters, cumulative_pct, alpha=0.15, color='#00d4ff')
        for pct, val, color in [
            (20, result.p20, '#ff6b6b'),
            (50, result.p50, '#ffd93d'),
            (80, result.p80, '#6bcb77'),
        ]:
            ax.axhline(pct, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
            ax.axvline(val, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
            ax.annotate(
                f'P{pct} = {val:.1f} cm',
                xy=(val, pct),
                xytext=(val + 0.5, pct + 3),
                color=color, fontsize=9, fontweight='bold',
            )
        ax.set_xlabel('Diámetro equivalente (cm)', color='white')
        ax.set_ylabel('Pasante acumulado (%)', color='white')
        ax.set_title('Curva Granulométrica — Análisis con SAM', color='white', fontsize=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(diameters) * 1.05)
        ax.grid(True, alpha=0.2, color='#555')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        buf.seek(0)
        return buf.read()

    @staticmethod
    def result_to_dict(result: AnalysisResult) -> dict:
        return {
            "image_path": result.image_path,
            "total_fragments": result.total_fragments,
            "scale_px_per_cm": round(result.scale_px_per_cm, 2),
            "granulometry": {
                "P20_cm": round(result.p20, 2),
                "P50_cm": round(result.p50, 2),
                "P80_cm": round(result.p80, 2),
                "mean_cm": round(result.mean_diameter, 2),
                "max_cm": round(result.max_diameter, 2),
                "min_cm": round(result.min_diameter, 2),
            },
            "fragments": [
                {
                    "id": f.id,
                    "diameter_cm": f.diameter_cm,
                    "area_cm2": f.area_cm2,
                    "circularity": f.circularity,
                }
                for f in result.fragments
            ],
        }
