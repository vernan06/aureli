"""Optional real-browser Aureli acceptance journey. Run against an isolated local server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from playwright.sync_api import sync_playwright, expect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--screenshots", type=Path, default=Path(".tmp/browser-screenshots"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    report: dict[str, object] = {"url": args.url, "journeys": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
        page.on("console", lambda message: errors.append(f"console error: {message.text}") if message.type == "error" else None)
        response = page.goto(args.url, wait_until="domcontentloaded")
        assert response and response.ok, f"App returned {response.status if response else 'no response'}"
        expect(page.get_by_role("heading", name="✦ Aureli")).to_be_visible()
        page.get_by_role("tab", name="Create account").click()
        page.get_by_label("Name", exact=True).fill("Acceptance Demo")
        page.get_by_label("Email address").fill(f"acceptance-{uuid.uuid4().hex[:10]}@example.test")
        page.get_by_label("Password (10+ characters)").fill("acceptance-only-password")
        page.get_by_role("button", name="Create account", exact=True).click()
        expect(page.get_by_role("heading", name="Your financial overview")).to_be_visible(timeout=30000)
        page.screenshot(path=str(args.screenshots / "desktop-overview-empty.png"), full_page=True)
        report["journeys"].append("account creation and empty overview")

        # Income and expenses
        page.get_by_test_id("stSidebarUserContent").get_by_text("Financial Planner", exact=True).click()
        page.get_by_test_id("stSidebarUserContent").get_by_text("Transactions", exact=True).click()
        transaction_form=page.get_by_test_id("stForm").filter(has=page.get_by_role("button", name="Add transaction"))
        transaction_form.get_by_role("combobox", name="Type").click()
        page.get_by_role("option", name="income", exact=True).click()
        transaction_form.get_by_label("Amount (₹)").fill("80000")
        transaction_form.get_by_label("Category", exact=True).fill("Salary")
        transaction_form.get_by_label("Description", exact=True).fill("Monthly salary")
        transaction_form.get_by_role("button", name="Add transaction").click()
        expect(page.get_by_text("Transaction saved.")).to_be_visible()
        transaction_form=page.get_by_test_id("stForm").filter(has=page.get_by_role("button", name="Add transaction"))
        transaction_form.get_by_role("combobox", name="Type").click()
        page.get_by_role("option", name="expense", exact=True).click()
        transaction_form.get_by_label("Amount (₹)").fill("10000")
        transaction_form.get_by_label("Category", exact=True).fill("Food")
        transaction_form.get_by_label("Description", exact=True).fill("Groceries")
        transaction_form.get_by_role("button", name="Add transaction").click()
        expect(page.get_by_text("Transaction saved.")).to_be_visible()
        report["journeys"].append("salary and expense transaction entry")

        # Budget and savings synchronization
        page.get_by_test_id("stSidebarUserContent").get_by_text("Budgets", exact=True).click()
        budget_form=page.get_by_test_id("stForm").filter(has=page.get_by_role("button", name="Save budget"))
        budget_form.get_by_label("Category", exact=True).fill("Food")
        budget_form.get_by_label("Monthly limit (₹)").fill("15000")
        budget_form.get_by_role("button", name="Save budget").click()
        expect(page.get_by_text("₹5,000.00 remaining")).to_be_visible()
        page.get_by_test_id("stSidebarUserContent").get_by_text("Savings Goals", exact=True).click()
        page.get_by_label("Goal name").fill("Emergency Fund")
        page.get_by_label("Target (₹)").fill("60000")
        page.get_by_label("Already saved (₹)").fill("5000")
        page.get_by_role("button", name="Create goal").click()
        expect(page.get_by_text("₹5,000.00", exact=False)).to_be_visible()
        page.get_by_text("Manage Emergency Fund", exact=True).click()
        page.get_by_label("Contribution (₹)").fill("10000")
        page.get_by_role("button", name="Record contribution").click()
        expect(page.get_by_text("₹15,000.00", exact=False)).to_be_visible()
        page.get_by_test_id("stSidebarUserContent").get_by_text("Overview", exact=True).click()
        expect(page.get_by_text("₹15,000.00", exact=False)).to_be_visible()
        report["journeys"].append("budget remainder and contribution-inclusive savings match on overview")

        # Loan cost preview, real payment defaults, and saved-loan scenario
        page.get_by_test_id("stSidebarUserContent").get_by_text("Loan Simulator", exact=True).click()
        page.get_by_label("Loan name").fill("Reference loan")
        page.get_by_label("Principal (₹)", exact=True).fill("100000")
        page.get_by_label("Annual interest (%)", exact=True).nth(0).fill("12")
        page.get_by_label("Duration (months)").fill("12")
        expect(page.get_by_text("₹8,884.88", exact=False)).to_be_visible()
        page.get_by_role("button", name="Save this loan").click()
        page.get_by_text("Reference loan", exact=False).click()
        expect(page.get_by_label("Total paid (₹)")).to_have_value("8884.88")
        expect(page.get_by_label("Principal component (₹)")).to_have_value("7884.88")
        page.get_by_role("button", name="Record payment").click()
        expect(page.get_by_text("Scheduled payment recorded successfully.")).to_be_visible()
        page.get_by_text("Reference loan", exact=False).click()
        expect(page.get_by_text("Recorded principal outstanding: ₹92,115.12", exact=False)).to_be_visible()
        expect(page.get_by_label("Baseline loan")).to_be_visible()
        expect(page.get_by_text("Interest saved")).to_be_visible()
        report["journeys"].append("loan preview, first-payment defaults, payment tracking and repayment comparison")

        # EMI inference (both real saved pipelines), then malformed-input feedback
        page.get_by_test_id("stSidebarUserContent").get_by_text("EMI Prediction", exact=True).click()
        salaries = page.get_by_label("Monthly Salary")
        expect(salaries).to_have_count(2)
        salaries.nth(0).fill("55000")
        page.get_by_role("button", name="Predict with saved Aureli model").nth(0).click()
        expect(page.get_by_text("Model assessment")).to_be_visible(timeout=60000)
        salaries=page.get_by_label("Monthly Salary")
        if not salaries.nth(1).is_visible(): page.get_by_text("Maximum affordable EMI", exact=True).click()
        salaries=page.get_by_label("Monthly Salary")
        salaries.nth(1).fill("55000")
        page.get_by_role("button", name="Predict with saved Aureli model").nth(1).click()
        expect(page.get_by_text("Estimated maximum monthly EMI")).to_be_visible(timeout=60000)
        page.screenshot(path=str(args.screenshots / "desktop-emi-predictions.png"), full_page=True)
        salaries = page.get_by_label("Monthly Salary")
        if not salaries.nth(0).is_visible(): page.get_by_text("EMI eligibility", exact=True).click()
        salaries=page.get_by_label("Monthly Salary")
        salaries.nth(0).fill("not-a-number")
        page.get_by_role("button", name="Predict with saved Aureli model").nth(0).click()
        expect(page.get_by_text("monthly_salary must be numeric.", exact=False)).to_be_visible()
        report["journeys"].append("two persisted model predictions and malformed-number error feedback")

        # Laboratory data and interactive charts
        page.get_by_test_id("stSidebarUserContent").get_by_text("Model Laboratory", exact=True).click()
        expect(page.get_by_text("Six-model comparison")).to_be_visible()
        expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=30000)
        page.screenshot(path=str(args.screenshots / "desktop-model-laboratory.png"), full_page=True)
        page.get_by_test_id("stSidebarUserContent").get_by_text("Overview", exact=True).click()
        expect(page.locator(".js-plotly-plot").first).to_be_visible()
        page.screenshot(path=str(args.screenshots / "desktop-overview-data.png"), full_page=True)
        report["journeys"].append("Model Laboratory metrics/plots and Overview cash-flow chart")

        # Responsive mobile layout on the authenticated session.
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(1200)
        width_check = page.evaluate("({viewport: window.innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth})")
        page.screenshot(path=str(args.screenshots / "mobile-overview.png"), full_page=True)
        report["mobile_width"] = width_check
        assert width_check["document"] <= width_check["viewport"] + 2, f"Mobile horizontal overflow: {width_check}"
        sidebar_control=page.locator('[data-testid="stSidebarCollapsedControl"]')
        if sidebar_control.count() and sidebar_control.is_visible(): sidebar_control.click()
        page.get_by_test_id("stSidebarUserContent").get_by_text("Financial Planner", exact=True).click()
        expect(page.get_by_text("Savings goals", exact=False)).to_be_visible()
        report["journeys"].append("mobile Overview sizing and opened-sidebar Financial Planner navigation")

        report["browser_errors"] = errors
        assert not errors, "Browser reported errors: " + json.dumps(errors, ensure_ascii=False)
        browser.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
