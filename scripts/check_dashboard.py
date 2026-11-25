"""Exercise the generated dashboard in an isolated local Chromium test browser."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/replay.html"))
    parser.add_argument("--screenshot", type=Path, default=Path("results/replay-screenshot.png"))
    parser.add_argument("--chromium", help="Optional path to an installed Chromium binary")
    args = parser.parse_args()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chromium, headless=True)
        page = browser.new_page(viewport={"width": 1152, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.input.resolve().as_uri())
        page.locator("#batches tr").first.wait_for()
        assert page.locator("#batches tr").count() == 20
        assert page.locator("#fpr").inner_text().endswith("%")
        page.locator("#policy").select_option("frozen")
        assert page.locator("#updates").inner_text() == "0"
        page.locator("#policy").select_option("gated_trim")
        assert int(page.locator("#updates").inner_text()) > 0
        page.locator("#scores").hover(position={"x": 173, "y": 80})
        assert page.locator("#detail").inner_text().startswith("Batch ")
        assert page.locator("#batches tr.active").count() == 1
        page.mouse.move(7, 7)
        page.screenshot(path=args.screenshot)
        page.locator("#scenario").select_option("unsw")
        assert "without timestamps" in page.locator("#description").inner_text()
        page.locator("#model").select_option("vae")
        assert page.locator("#policy option").count() == 2
        page.set_viewport_size({"width": 389, "height": 844})
        assert page.locator("#scores").bounding_box()["width"] < 389
        assert not errors, errors
        browser.close()
    print("Dashboard checks passed (selectors, chart interaction, and mobile layout).")


if __name__ == "__main__":
    main()
