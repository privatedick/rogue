"""System health check and capability analyzer."""
"""
"""
This module tests various components of the system and provides
recommendations for what operations are currently possible based
on which components are functioning.
"""
"""
"""
"""
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

        try:
        from dotenv import load_dotenv
        from google import generativeai as genai
        from rich.console import Console
        from rich.table import Table

        HAS_RICH = True
            except ImportError:
            HAS_RICH = False


            # Basic test result container
            @dataclass
                class TestResult:
                """Contains the result of a system component test."""

                name: str
            passed: bool
            error: str = ""
            details: Dict[str, Any] = None


                class SystemHealthCheck:
                """System health checker and capability analyzer."""

                    def __init__(self):
                    """Initialize the health checker."""
                    self.console = Console() if HAS_RICH else None
                    self.results: List[TestResult] = []

                    async def run_all_tests(self) -> List[TestResult]:
                    """Run all system health checks."""
                    """
                    """
                    Returns:
                    List of test results
                    """
                    """
                    tests = [
                    self.test_environment,
                    self.test_gemini_access,
                    self.test_project_structure,
                    self.test_permissions,
                    ]"""
                    """

                        for test in tests:
                            try:
                            result = await test()
                            self.results.append(result)
                                except Exception as e:
                                self.results.append(
                                TestResult(name=test.__name__, passed=False, error=str(e))
                                )

                            return self.results

                            async def test_environment(self) -> TestResult:
                            """Test environment configuration."""
                            load_dotenv()
                            missing_vars = []

                            # Required environment variables
                            required_vars = ["GEMINI_API_KEY"]

                                for var in required_vars:
                                    if not os.getenv(var):
                                    missing_vars.append(var)

                                return TestResult(
                                name="Environment Configuration",
                            passed=len(missing_vars) == 0,
                            error=f"Missing environment variables: {', '.join(missing_vars)}"
                            if missing_vars
                            else "",
                            details={"missing_vars": missing_vars},
                            )

                            async def test_gemini_access(self) -> TestResult:
                            """Test Gemini API access."""
                                try:
                                api_key = os.getenv("GEMINI_API_KEY")
                                    if not api_key:
                                return TestResult(
                                name="Gemini API Access", passed=False, error="No API key found"
                                )

                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-1219")

                                # Simple test prompt
                                response = await model.generate_content("Say 'test' if you can read this.")

                            return TestResult(
                            name="Gemini API Access",
                        passed="test" in response.text.lower(),
                        details={"response": response.text},
                        )
                            except Exception as e:
                        return TestResult(name="Gemini API Access", passed=False, error=str(e))

                        async def test_project_structure(self) -> TestResult:
                        """Test project directory structure."""
                        required_dirs = [
                        "src",
                        "tests",
                        "logs",
                        "output",
                        "prompts",
                        ]

                        missing_dirs = []
                        root = Path.cwd()

                            for dir_name in required_dirs:
                                if not (root / dir_name).exists():
                                missing_dirs.append(dir_name)

                            return TestResult(
                            name="Project Structure",
                        passed=len(missing_dirs) == 0,
                        error=f"Missing directories: {', '.join(missing_dirs)}"
                        if missing_dirs
                        else "",
                        details={"missing_dirs": missing_dirs},
                        )

                        async def test_permissions(self) -> TestResult:
                        """Test file system permissions."""
                        test_dirs = ["logs", "output"]
                        permission_errors = []

                            for dir_name in test_dirs:
                            test_file = Path(dir_name) / f"test_{datetime.now().timestamp()}.txt"
                                try:
                                test_file.parent.mkdir(exist_ok=True)
                                test_file.write_text("test")
                                test_file.unlink()
                                    except Exception as e:
                                    permission_errors.append(f"{dir_name}: {str(e)}")

                                return TestResult(
                                name="File System Permissions",
                            passed=len(permission_errors) == 0,
                            error="\n".join(permission_errors) if permission_errors else "",
                            details={"permission_errors": permission_errors},
                            )

                                def get_capabilities(self) -> List[str]:
                                """Determine what operations are possible based on test results."""
                                """
                                """
                                Returns:
                                List of possible operations
                                """
                                """
                                capabilities = []"""
                                """

                                # Check basic Gemini operations
                                if all(
                                r.passed
                                for r in self.results
                                if r.name in ["Environment Configuration", "Gemini API Access"]
                                ):
                                capabilities.extend(
                                [
                                "- Kan köra direkta Gemini-förfrågningar",
                                "- Kan generera kod och förslag",
                                "- Kan utföra kodanalys",
                                ]
                                )

                                # Check file operations
                                if all(
                                r.passed
                                for r in self.results
                                if r.name in ["Project Structure", "File System Permissions"]
                                ):
                                capabilities.extend(
                                [
                                "- Kan spara och läsa filer",
                                "- Kan generera och modifiera kod i projekt",
                                "- Kan skapa loggar och output",
                                ]
                                )

                            return capabilities

                                def get_recommendations(self) -> List[str]:
                                """Get recommendations for fixing issues."""
                                """
                                """
                                Returns:
                                List of recommendations
                                """
                                """
                                recommendations = []"""
                                """

                                    for result in self.results:
                                        if not result.passed:
                                            if result.name == "Environment Configuration":
                                            recommendations.append(
                                            "Sätt upp saknade miljövariabler i .env-filen: "
                                            + ", ".join(result.details.get("missing_vars", []))
                                            )

                                                elif result.name == "Gemini API Access":
                                                recommendations.append(
                                                "Kontrollera att API-nyckeln är giltig och att du har "
                                                "internetåtkomst"
                                                )

                                                    elif result.name == "Project Structure":
                                                    recommendations.append(
                                                    "Skapa saknade projektkataloger: "
                                                    + ", ".join(result.details.get("missing_dirs", []))
                                                    )

                                                        elif result.name == "File System Permissions":
                                                        recommendations.append(
                                                        "Åtgärda behörighetsproblem för kataloger: "
                                                        + ", ".join(result.details.get("permission_errors", []))
                                                        )

                                                    return recommendations

                                                        def display_results(self):
                                                        """Display test results, capabilities, and recommendations."""
                                                            if self.console:
                                                            # Create results table
                                                            table = Table(title="Systemhälsotest Resultat")
                                                            table.add_column("Komponent")
                                                            table.add_column("Status")
                                                            table.add_column("Felmeddelande", style="red")

                                                                for result in self.results:
                                                                table.add_row(
                                                                result.name,
                                                                "[green]OK[/green]" if result.passed else "[red]FEL[/red]",
                                                                result.error,
                                                                )

                                                                self.console.print(table)

                                                                # Display capabilities
                                                                self.console.print("\n[bold green]Möjliga operationer:[/bold green]")
                                                                    for cap in self.get_capabilities():
                                                                    self.console.print(cap)

                                                                    # Display recommendations
                                                                    recommendations = self.get_recommendations()
                                                                        if recommendations:
                                                                        self.console.print("\n[bold yellow]Rekommendationer:[/bold yellow]")
                                                                            for rec in recommendations:
                                                                            self.console.print(f"- {rec}")
                                                                                else:
                                                                                # Simple text output
                                                                                print("\nSystemhälsotest Resultat:")
                                                                                    for result in self.results:
                                                                                    status = "OK" if result.passed else "FEL"
                                                                                    print(f"{result.name}: {status}")
                                                                                        if result.error:
                                                                                        print(f"  Fel: {result.error}")

                                                                                        print("\nMöjliga operationer:")
                                                                                            for cap in self.get_capabilities():
                                                                                            print(cap)

                                                                                            recommendations = self.get_recommendations()
                                                                                                if recommendations:
                                                                                                print("\nRekommendationer:")
                                                                                                    for rec in recommendations:
                                                                                                    print(f"- {rec}")


                                                                                                    async def main():
                                                                                                    """Run health check from command line."""
                                                                                                    checker = SystemHealthCheck()
                                                                                                    await checker.run_all_tests()
                                                                                                    checker.display_results()


                                                                                                        if __name__ == "__main__":
                                                                                                        asyncio.run(main())
