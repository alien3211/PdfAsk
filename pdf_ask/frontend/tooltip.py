import logging

logger = logging.getLogger(__name__)

TOOLTIP_CSS = """
<style>
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            font-size: smaller;
            vertical-align: super;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position the tooltip above the text */
            left: 50%;
            transform: translateX(-50%); /* Center the tooltip */
            opacity: 0;
            transition: opacity 0.3s;
            white-space: normal; /* Prevent text from wrapping */
            max-width: 600px; /* Ensure tooltip does not exceed viewport width */
            overflow: visible;
            word-wrap: break-word;
            width: max-content; /* Ensure tooltip expands to fit content */

        }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
"""


def add_tooltip_css_to_markdown(markdown_text: str) -> str:
    """Adds the tooltip CSS to a markdown string.

    Args:
        markdown_text: str

    Returns:
        str: Markdown text with embedded tooltip CSS
    """
    return f"""
{TOOLTIP_CSS}
{markdown_text}
"""


def create_tooltip_span(text: str, tooltip_text: str) -> str:
    """Creates a span element with tooltip styling.

    Args:
        text: str
        tooltip_text: str

    Returns:
        str: HTML span element with tooltip
    """
    logger.debug(f"Creating tooltip span for text: {text} with tooltip: {tooltip_text}")
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'


def replace_text_with_tooltips(text: str, tooltip_map: dict[str, str]) -> str:
    """Replaces specified items in the text with tooltips.

    Args:
        text: str - the text to add the tooltip to
        tooltip_map: dict[str, str] - a mapping of items to their tooltip text

    Returns:
        str - the text with the appropriate tooltip spans
    """
    logger.debug("Starting text replacement with tooltips.")
    for item, tooltip in tooltip_map.items():
        logger.debug(f"Replacing '{item}' with tooltip.")
        text = text.replace(item, create_tooltip_span(item, tooltip))
    logger.debug("Finished replacing text with tooltips.")
    return add_tooltip_css_to_markdown(text)
