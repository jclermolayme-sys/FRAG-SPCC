"""
RockFrag Core - Motor de segmentación con SAM (Segment Anything Model)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass, field
from typing import Optional, List
from ultralytics import SAM

# ─── Estructuras de datos (igual que antes) ────────────────────────────────────

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

# ─── Motor con SAM ─────────────────────────────────────────────────────────

class RockFragAnalyzer:
    def __init__(
        self,
        scale_reference_cm: float = 30.0,
        min_fragment_area_px: int = 200,
        max_fragment_ratio: float = 0.8,
        sam_model_path: str = "sam2_t.pt",   # puedes usar sam2_s.pt, sam2_b.pt, etc.
        sam_confidence: float = 0.5,          # no usado directamente, pero por compatibilidad
    ):
        self.scale_reference_cm = scale_reference_cm
        self.min_fragment_area_px = min_fragment_area_px
        self.max_fragment_ratio = max_fragment_ratio
        self.sam_model_path = sam_model_path
        self.sam = None  # se cargará bajo demanda

    def _load_sam(self):
        if self.sam is None:
            print(f"Cargando modelo SAM desde {self.sam_model_path}...")
            self.sam = SAM(self.sam_model_path)
            # Opcional: mover a GPU si está disponible
            if torch.cuda.is_available():
                self.sam.to('cuda')

    def analyze(
        self,
        image_path: str,
        scale_px_per_cm: Optional[float] = None,
        use_watershed: bool = True,   # mantenemos para compatibilidad, pero ignoramos
    ) -> AnalysisResult:
        """
        Pipeline con SAM.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {image_path}")

        h, w = img.shape[:2]

        # Determinar escala
        if scale_px_per_cm is None:
            scale_px_per_cm = self.detect_scale_bar(img)
            if scale_px_per_cm is None:
                scale_px_per_cm = (w * 0.10) / self.scale_reference_cm

        # Cargar SAM si es necesario
        self._load_sam()

        # Convertir a RGB (SAM espera RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ejecutar SAM sobre la imagen completa (sin prompts)
        results = self.sam(img_rgb)
        masks = []
        if results[0].masks is not None:
            masks_tensor = results[0].masks.data.cpu().numpy()
            for mask in masks_tensor:
                area = np.sum(mask)
                if area >= self.min_fragment_area_px:
                    masks.append(mask.astype(bool))

        # Convertir máscaras a contornos y calcular métricas
        fragments = []
        for i, mask in enumerate(masks):
            # Convertir máscara booleana a imagen uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_fragment_area_px:
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
            raise ValueError("No se detectaron fragmentos. Verifica la imagen o ajusta parámetros.")

        fragments_sorted = sorted(fragments, key=lambda f: f.diameter_cm)
        diameters = [f.diameter_cm for f in fragments_sorted]

        result = AnalysisResult(
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
        return result

    # Mantenemos el método detect_scale_bar (igual que antes)
    def detect_scale_bar(self, img: np.ndarray) -> Optional[float]:
        # ... (código igual al que tenías)
        pass
