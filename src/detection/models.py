"""Data models for detection results."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates (normalized 0-1)."""

    x_min: float = Field(ge=0, le=1, description="Left edge (0-1)")
    y_min: float = Field(ge=0, le=1, description="Top edge (0-1)")
    x_max: float = Field(ge=0, le=1, description="Right edge (0-1)")
    y_max: float = Field(ge=0, le=1, description="Bottom edge (0-1)")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_pixel_coords(self, image_width: int, image_height: int) -> dict[str, int]:
        """Convert normalized coords to pixel coordinates."""
        return {
            "x_min": int(self.x_min * image_width),
            "y_min": int(self.y_min * image_height),
            "x_max": int(self.x_max * image_width),
            "y_max": int(self.y_max * image_height),
        }


class Detection(BaseModel):
    """A single detected weed instance."""

    label: str = Field(description="Weed type label (e.g., 'dandelion')")
    confidence: float = Field(ge=0, le=1, description="Detection confidence score")
    box: BoundingBox = Field(description="Bounding box location")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "dandelion",
                "confidence": 0.87,
                "box": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.5},
            }
        }


class DetectionResult(BaseModel):
    """Complete detection result for an image."""

    detections: list[Detection] = Field(default_factory=list)
    image_width: int = Field(description="Original image width in pixels")
    image_height: int = Field(description="Original image height in pixels")
    inference_time_ms: float = Field(description="Time taken for inference")

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Return a new result with only detections above the threshold."""
        filtered = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
        )

    def filter_by_size(
        self, min_size: float = 0.005, max_size: float = 0.25
    ) -> "DetectionResult":
        """
        Filter detections by bounding box area.

        Removes boxes that are too small (noise) or too large (false positives
        that cover too much of the image to be a single weed).

        Args:
            min_size: Minimum box area as fraction of image (0-1). Default 0.5%
            max_size: Maximum box area as fraction of image (0-1). Default 25%

        Returns:
            DetectionResult with size-filtered detections
        """
        filtered = []
        for d in self.detections:
            area = d.box.width * d.box.height
            if min_size <= area <= max_size:
                filtered.append(d)
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
        )

    def filter_adaptive(
        self,
        base_threshold: float = 0.1,
        min_threshold: float = 0.05,
        max_threshold: float = 0.4,
        target_density: float = 0.02,
    ) -> "DetectionResult":
        """
        Adaptive threshold filtering based on detection density.

        Adjusts threshold dynamically:
        - Few detections? Lower threshold to catch sparse weeds
        - Many detections? Raise threshold to reduce false positives

        Args:
            base_threshold: Starting confidence threshold
            min_threshold: Never go below this (too many false positives)
            max_threshold: Never go above this (might miss real weeds)
            target_density: Target detections per unit image area (0-1 normalized)

        Returns:
            Filtered DetectionResult with adaptive threshold applied
        """
        # Calculate detection density (detections per normalized unit area)
        image_area = self.image_width * self.image_height
        current_density = self.count / image_area if image_area > 0 else 0

        # Calculate density ratio - how far off from target we are
        if target_density > 0:
            density_ratio = current_density / target_density
        else:
            density_ratio = 1.0

        # Adjust threshold based on density ratio using log scale for smoother response
        # ratio > 1 means too many detections -> raise threshold
        # ratio < 1 means too few detections -> lower threshold
        import math
        adjustment = math.log2(max(density_ratio, 0.1)) * 0.05  # 0.05 scaling factor
        effective_threshold = base_threshold + adjustment

        # Clamp to min/max bounds
        effective_threshold = max(min_threshold, min(max_threshold, effective_threshold))

        filtered = [d for d in self.detections if d.confidence >= effective_threshold]
        return DetectionResult(
            detections=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
        )

    def get_counts_by_label(self) -> dict[str, int]:
        """Count detections by weed type."""
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return counts

    def deduplicate(self, min_distance: float = 0.05) -> "DetectionResult":
        """
        Remove duplicate detections using center-distance clustering.

        When multiple reference images detect the same weed, we get many
        overlapping boxes. This keeps only the highest-confidence detection
        when box centers are close together.

        Args:
            min_distance: Minimum distance (0-1 normalized) between box centers
                          to consider them separate detections. Default 0.05 = 5% of image.

        Returns:
            DetectionResult with duplicates removed
        """
        if not self.detections:
            return self

        # Sort by confidence descending - highest confidence kept first
        sorted_dets = sorted(self.detections, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []

        for det in sorted_dets:
            # Check if this detection is too close to any kept detection
            is_duplicate = False
            det_center = det.box.center

            for kept in keep:
                # Only compare same label
                if det.label != kept.label:
                    continue

                kept_center = kept.box.center
                # Euclidean distance between centers (normalized 0-1 space)
                distance = (
                    (det_center[0] - kept_center[0]) ** 2 +
                    (det_center[1] - kept_center[1]) ** 2
                ) ** 0.5

                if distance < min_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(det)

        return DetectionResult(
            detections=keep,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
        )

    def cluster_overlapping(self) -> "DetectionResult":
        """
        Merge overlapping detections into single bounding boxes.

        Finds boxes that overlap and merges them into a single larger box
        covering the entire weed region. This groups individual leaf/element
        detections into unified weed regions.

        Returns:
            DetectionResult with merged boxes
        """
        if not self.detections:
            return self

        def boxes_overlap(box1: BoundingBox, box2: BoundingBox) -> bool:
            """Check if boxes overlap at all (any IoU > 0)."""
            return not (
                box1.x_max < box2.x_min or box2.x_max < box1.x_min or
                box1.y_max < box2.y_min or box2.y_max < box1.y_min
            )

        # Group detections by label
        by_label: dict[str, list[Detection]] = {}
        for det in self.detections:
            by_label.setdefault(det.label, []).append(det)

        merged_detections: list[Detection] = []

        for label, dets in by_label.items():
            # Sort by confidence descending
            dets = sorted(dets, key=lambda d: d.confidence, reverse=True)

            # Union-Find style clustering: merge overlapping boxes
            clusters: list[list[Detection]] = []

            for det in dets:
                merged_into = None
                for cluster in clusters:
                    # Check if this detection overlaps with any in the cluster
                    for existing in cluster:
                        if boxes_overlap(det.box, existing.box):
                            cluster.append(det)
                            merged_into = cluster
                            break
                    if merged_into:
                        break

                if not merged_into:
                    clusters.append([det])

            # Merge clusters that now overlap due to additions
            changed = True
            while changed:
                changed = False
                new_clusters = []
                for cluster in clusters:
                    merged = False
                    for new_cluster in new_clusters:
                        # Check if any boxes between clusters overlap
                        for d1 in cluster:
                            for d2 in new_cluster:
                                if boxes_overlap(d1.box, d2.box):
                                    new_cluster.extend(cluster)
                                    merged = True
                                    changed = True
                                    break
                            if merged:
                                break
                        if merged:
                            break
                    if not merged:
                        new_clusters.append(cluster)
                clusters = new_clusters

            # Create merged detection for each cluster
            for cluster in clusters:
                # Compute bounding box that covers all detections
                x_min = min(d.box.x_min for d in cluster)
                y_min = min(d.box.y_min for d in cluster)
                x_max = max(d.box.x_max for d in cluster)
                y_max = max(d.box.y_max for d in cluster)

                # Use highest confidence from the cluster
                best_conf = max(d.confidence for d in cluster)

                merged_detections.append(Detection(
                    label=label,
                    confidence=best_conf,
                    box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                ))

        # Sort by confidence descending
        merged_detections.sort(key=lambda d: d.confidence, reverse=True)

        return DetectionResult(
            detections=merged_detections,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
        )
