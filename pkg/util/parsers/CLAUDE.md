# Package: parsers

## Purpose
Provides parsing utilities for container images and cron schedules.

## Key Functions
- `ParseImageName()` - Parses a Docker image string into repo, tag, and digest components
- `ParseCronScheduleWithPanicRecovery()` - Safely parses cron schedules, recovering from panics

## ParseImageName Details
- Input: "registry.io/repo/image:tag@sha256:abc..."
- Output: repository name, tag, digest (any can be empty)
- Defaults to "latest" tag if neither tag nor digest specified
- Uses docker/distribution reference parsing

## ParseCronScheduleWithPanicRecovery Details
- Wraps cron.ParseStandard with panic recovery
- Handles malformed schedules like "TZ=0" that can cause panics
- Returns error instead of panicking

## Design Patterns
- Imports crypto/sha256 and sha512 for digest support
- Uses defer/recover for panic-safe parsing
- Normalizes image names to canonical form
