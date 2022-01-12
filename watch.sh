#!/usr/bin/env bash
watch "kubectl describe pod | grep -A20 Events"
