#!/bin/bash

#node06
for node in node08 node; do
	command -v ${node} > /dev/null 2>&1 || continue

	echo "Testing with $(${node} --version)..."
	${node} node_modules/vows/bin/vows test/*test.js
done
