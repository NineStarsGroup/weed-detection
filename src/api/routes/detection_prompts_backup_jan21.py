"""
Backup of DINO text prompts - January 21, 2026

This file preserves the original DINO prompts before the leaf-focused optimization.
To rollback: copy the _build_text_queries_backup function content back to detection.py

Key difference: This version includes flower-based prompts for some weed types.
The optimized version focuses entirely on leaf morphology for better year-round detection.
"""


def _build_text_queries_backup(weed_types: list[str], for_dino: bool = False) -> list[str]:
    """
    Build text queries for text-guided detection models.

    BACKUP VERSION - Jan 21, 2026
    Contains original prompts before leaf-focused optimization.

    Args:
        weed_types: List of weed types to detect
        for_dino: If True, use fewer but more targeted prompts for DINO
                  to reduce overlapping detections
    """
    text_queries = []
    for weed_type in weed_types:
        if weed_type.lower() == "dandelion":
            if for_dino:
                # DINO: Focus on leaves only - detect plants even without flowers
                text_queries.extend([
                    "dandelion plant with jagged leaves",    # Leaf rosettes (primary)
                    "rosette of serrated green leaves",      # Leaf pattern description
                    "dandelion leaves growing from center",  # Radial leaf pattern
                ])
            else:
                # OWLv2: simpler prompts work better
                text_queries.extend([
                    "dandelion",
                    "yellow dandelion flower",
                    "dandelion puffball",
                    "dandelion leaves",
                ])

        elif weed_type.lower() == "clover":
            if for_dino:
                # Focus on the distinctive three-leaf pattern
                text_queries.extend([
                    "clover with three round leaves",        # Trifoliate pattern (primary)
                    "three leaflets in clover pattern",      # Leaf structure
                    "clover leaf with three rounded lobes",  # Individual leaf detail
                ])
            else:
                text_queries.extend([
                    "clover",
                    "three leaf clover",
                    "clover leaves",
                    "trifoliate clover",
                ])

        elif weed_type.lower() == "crabgrass":
            if for_dino:
                # Focus on the distinctive spreading grass pattern
                text_queries.extend([
                    "crabgrass with spreading stems",    # Leaf/stem pattern
                    "low growing grass weed",            # Growth habit
                ])
            else:
                text_queries.extend([
                    "crabgrass",
                    "crabgrass weed",
                ])

        elif weed_type.lower() == "poa_annua":
            if for_dino:
                # Poa annua (annual bluegrass) - taller clumpy grass that stands above turf
                text_queries.extend([
                    "tall grass clump standing above lawn",     # Height difference (primary)
                    "raised grass tuft in mowed lawn",          # Clumpy growth habit
                    "grass clump taller than surrounding turf", # Relative height
                ])
            else:
                text_queries.extend([
                    "poa annua",
                    "annual bluegrass",
                    "tall grass clump",
                ])

        elif weed_type.lower() == "silverleaf_nightshade":
            if for_dino:
                # Silverleaf nightshade - silvery leaves with purple/white flowers
                # NOTE: This version includes flower prompt
                text_queries.extend([
                    "silverleaf nightshade plant with silvery leaves",
                    "nightshade with star-shaped purple flowers",
                    "plant with silver-gray fuzzy leaves",
                ])
            else:
                text_queries.extend([
                    "silverleaf nightshade",
                    "nightshade plant",
                    "silvery weed leaves",
                ])

        elif weed_type.lower() == "field_bindweed":
            if for_dino:
                # Field bindweed - vine with white/pink trumpet flowers
                # NOTE: This version includes flower prompts
                text_queries.extend([
                    "bindweed vine with trumpet-shaped flowers",
                    "morning glory weed with arrow-shaped leaves",
                    "creeping vine with white or pink flowers",
                ])
            else:
                text_queries.extend([
                    "field bindweed",
                    "bindweed vine",
                    "morning glory weed",
                ])

        elif weed_type.lower() == "broom_snakeweed":
            if for_dino:
                # Broom snakeweed - woody shrub with yellow flowers
                # NOTE: This version includes flower prompts
                text_queries.extend([
                    "broom snakeweed with yellow flower clusters",
                    "woody shrub with small yellow flowers",
                    "snakeweed plant with thin green stems",
                ])
            else:
                text_queries.extend([
                    "broom snakeweed",
                    "snakeweed shrub",
                    "yellow flowering weed",
                ])

        elif weed_type.lower() == "palmers_amaranth":
            if for_dino:
                # Palmer's amaranth - tall broadleaf with long seed head
                # NOTE: This version includes seed head/flower prompts
                text_queries.extend([
                    "palmer amaranth with tall seed spike",
                    "pigweed with long reddish flower spike",
                    "broadleaf weed with elongated seed head",
                ])
            else:
                text_queries.extend([
                    "palmer amaranth",
                    "pigweed",
                    "amaranth weed",
                ])

        elif weed_type.lower() == "russian_thistle":
            if for_dino:
                # Russian thistle (tumbleweed) - spiny round plant
                text_queries.extend([
                    "russian thistle with spiny branches",
                    "tumbleweed plant with thin needle-like leaves",
                    "round bushy weed with spine-tipped stems",
                ])
            else:
                text_queries.extend([
                    "russian thistle",
                    "tumbleweed",
                    "prickly weed",
                ])

        else:
            text_queries.append(weed_type)
            if not for_dino:
                text_queries.append(f"{weed_type} plant")
    return text_queries
