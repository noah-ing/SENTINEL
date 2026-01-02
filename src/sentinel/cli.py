"""Command-line interface for SENTINEL."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sentinel.detection.pipeline import SentinelDetector, DetectorConfig
from sentinel.evaluation.benchmark import SentinelBenchmark, load_attacks


def main():
    """Main entry point for the SENTINEL CLI."""
    parser = argparse.ArgumentParser(
        description="SENTINEL: Prompt Injection Detection for Agentic AI Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentinel scan "Ignore previous instructions"
  sentinel scan --file document.txt --depth thorough
  sentinel benchmark --suite quick
  sentinel benchmark --suite full --output results.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan content for injection")
    scan_parser.add_argument("content", nargs="?", help="Content to scan")
    scan_parser.add_argument("--file", "-f", help="File to scan")
    scan_parser.add_argument(
        "--depth",
        choices=["fast", "adaptive", "thorough"],
        default="adaptive",
        help="Detection depth (default: adaptive)",
    )
    scan_parser.add_argument(
        "--task",
        default="general",
        help="Task context for detection",
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark suite to run (default: quick)",
    )
    bench_parser.add_argument(
        "--output", "-o",
        help="Output file for detailed results (JSON)",
    )
    bench_parser.add_argument(
        "--depth",
        choices=["fast", "adaptive", "thorough"],
        default="adaptive",
        help="Detection depth (default: adaptive)",
    )

    # List attacks command
    list_parser = subparsers.add_parser("list-attacks", help="List available attacks")
    list_parser.add_argument(
        "--category",
        help="Filter by category",
    )
    list_parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "expert"],
        help="Filter by difficulty",
    )

    # Red team command
    redteam_parser = subparsers.add_parser("redteam", help="Run adversarial red team session")
    redteam_parser.add_argument(
        "--generate", "-g",
        type=int,
        default=10,
        help="Number of novel attacks to generate (default: 10)",
    )
    redteam_parser.add_argument(
        "--mutate", "-m",
        type=int,
        default=20,
        help="Number of mutation variants to create (default: 20)",
    )
    redteam_parser.add_argument(
        "--output", "-o",
        help="Output file for bypasses (JSON)",
    )
    redteam_parser.add_argument(
        "--category",
        help="Focus on specific attack category",
    )

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "scan":
        asyncio.run(run_scan(args))
    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args))
    elif args.command == "redteam":
        asyncio.run(run_redteam(args))
    elif args.command == "list-attacks":
        run_list_attacks(args)
    elif args.command == "version":
        from sentinel import __version__
        print(f"SENTINEL v{__version__}")
    else:
        parser.print_help()
        sys.exit(1)


async def run_scan(args):
    """Run the scan command."""
    # Get content
    if args.file:
        with open(args.file) as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        # Read from stdin
        content = sys.stdin.read()

    if not content.strip():
        print("Error: No content to scan", file=sys.stderr)
        sys.exit(1)

    # Create detector
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))

    # Scan
    result = await detector.scan(
        content,
        context={"task": args.task, "tools": []},
        depth=args.depth,
    )

    # Output
    if args.json:
        output = {
            "is_injection": result.is_injection,
            "confidence": result.confidence,
            "layer": result.layer,
            "details": result.details,
            "latency_ms": result.latency_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        status = "DETECTED" if result.is_injection else "CLEAN"
        print(f"Status: {status}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Layer: {result.layer}")
        print(f"Latency: {result.latency_ms:.2f}ms")

        if result.details:
            print(f"Details: {result.details}")

    sys.exit(1 if result.is_injection else 0)


async def run_benchmark(args):
    """Run the benchmark command."""
    print(f"Running SENTINEL benchmark (suite: {args.suite})")
    print("-" * 50)

    # Create detector
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))

    # Create benchmark
    benchmark = SentinelBenchmark(attack_suite=args.suite)

    print(f"Loaded {len(benchmark.attacks)} attacks")
    print()

    # Run benchmark
    results = await benchmark.run(
        detector=detector,
        depth=args.depth,
        show_progress=True,
    )

    # Print results
    print()
    print(results.summary())

    # Export if requested
    if args.output:
        benchmark.export_results(Path(args.output))
        print(f"\nDetailed results exported to: {args.output}")


def run_list_attacks(args):
    """List available attacks."""
    attacks = load_attacks("full")

    # Filter
    if args.category:
        attacks = [a for a in attacks if args.category in a.category]
    if args.difficulty:
        attacks = [a for a in attacks if a.difficulty == args.difficulty]

    # Group by category
    by_category = {}
    for attack in attacks:
        if attack.category not in by_category:
            by_category[attack.category] = []
        by_category[attack.category].append(attack)

    # Print
    print(f"Available attacks: {len(attacks)}")
    print("-" * 50)

    for category in sorted(by_category.keys()):
        category_attacks = by_category[category]
        print(f"\n{category} ({len(category_attacks)} attacks):")

        for attack in category_attacks:
            difficulty_badge = {
                "easy": "[E]",
                "medium": "[M]",
                "hard": "[H]",
                "expert": "[X]",
            }.get(attack.difficulty, "[?]")

            print(f"  {difficulty_badge} {attack.id}: {attack.name}")


async def run_redteam(args):
    """Run a red team session."""
    from sentinel.redteam.loop import RedTeamLoop
    from sentinel.redteam.generator import MockAttackGenerator

    print("SENTINEL Red Team Session")
    print("=" * 60)
    print(f"Generating {args.generate} novel attacks")
    print(f"Creating {args.mutate} mutation variants")
    if args.category:
        print(f"Focusing on category: {args.category}")
    print()

    # Create components
    detector = SentinelDetector(DetectorConfig(use_mock_models=True))
    generator = MockAttackGenerator()

    loop = RedTeamLoop(detector=detector, generator=generator)

    # Callback for bypasses
    def on_bypass(bypass):
        print(f"  [BYPASS] {bypass.attack.name}")
        print(f"           Confidence: {bypass.detection_confidence:.2f}")
        print(f"           Type: {bypass.bypass_type}")

    # Run session
    categories = [args.category] if args.category else None

    result = await loop.run(
        num_generated=args.generate,
        num_mutations=args.mutate,
        categories=categories,
        on_bypass=on_bypass,
    )

    # Print results
    print()
    print(result.summary())

    # Show weakest categories
    weak = loop.get_weakest_categories(3)
    if weak:
        print("\nWeakest Categories (highest bypass rate):")
        for cat, rate in weak:
            print(f"  {cat}: {rate:.1%}")

    # Export bypasses
    if args.output:
        loop.export_bypasses(args.output)
        print(f"\nBypasses exported to: {args.output}")
    elif result.bypasses_found > 0:
        print(f"\nUse --output to export {result.bypasses_found} bypasses for analysis")


if __name__ == "__main__":
    main()
