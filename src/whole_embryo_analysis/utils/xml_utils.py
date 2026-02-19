import xml.etree.ElementTree as ET
from pathlib import Path


def get_channels(xml_file: str) -> list[int]:
    """Parse a BigStitcher XML to determine available channel IDs.

    Tries common BigStitcher XML paths for channel IDs and returns a sorted list.
    Falls back to [0] if nothing can be detected.
    """
    try:
        xml_path = Path(xml_file)
        if not xml_path.exists():
            return [0]

        root = ET.parse(str(xml_path)).getroot()

        channel_ids: set[int] = set()

        # Typical BigStitcher path
        for node in root.findall(".//ViewSetup/channel/id"):
            if node.text is not None:
                try:
                    channel_ids.add(int(node.text))
                except ValueError:
                    continue

        # Alternate path seen in some exports
        if not channel_ids:
            for node in root.findall(".//Channel/id"):
                if node.text is not None:
                    try:
                        channel_ids.add(int(node.text))
                    except ValueError:
                        continue

        return sorted(channel_ids) if channel_ids else [0]
    except Exception:
        return [0]


def check_xml_stage(xml_file: str, stage: str) -> bool:
    """Check whether an XML has results for a specific processing stage.

    stage == "pairwise": looks for PairwiseResult elements
    stage == "solver": looks for InterpolatedAffineModel3D registrations
    """
    try:
        root = ET.parse(xml_file).getroot()
        if stage == "pairwise":
            return len(root.findall(".//PairwiseResult")) > 0
        if stage == "solver":
            return len(root.findall(".//InterpolatedAffineModel3D")) > 0
        return False
    except Exception:
        return False


