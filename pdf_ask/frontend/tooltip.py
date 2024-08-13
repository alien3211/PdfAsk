TOOLTIP_STYLE = """
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


def _add_tooltip_style_to_markdown(text: str) -> str:
    """Adds the tooltip style to a markdown string.

    Args:
        text
    """
    return f"""
{TOOLTIP_STYLE}
{text}
"""


def _add_tooltip_style_to_text(text: str, tooltip_text: str) -> str:
    """Adds the tooltip style to a text string.

    Args:
        text: str
        tooltip_text: str
    """
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'


def replace_with_tooltips(text: str, tooltip_map: dict[str, str]) -> str:
    """Replaces the given items in the text with tooltips.

    Args:
        text: str - the text to add the tooltip to
        tooltip_map: dict[str, str] - a mapping of items to their tooltip text

    Returns:
        str - the text with the appropriate tooltip spans
    """
    for item, tooltip in tooltip_map.items():
        text = text.replace(item, _add_tooltip_style_to_text(item, tooltip))
    return _add_tooltip_style_to_markdown(text)
