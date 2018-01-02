
all:
	swift build
	.build/debug/TemplateEncoder > Sources/gnostic-swift-generator/Templates.swift
	swift build
	cp .build/debug/gnostic-swift-generator gnostic-swift-generator
	rm -f gnostic-swift-client gnostic-swift-server
	ln -s gnostic-swift-generator gnostic-swift-client
	ln -s gnostic-swift-generator gnostic-swift-server

clean:
	rm -rf .build Packages
	rm -rf gnostic-swift-client gnostic-swift-service
