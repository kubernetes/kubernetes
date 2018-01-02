
all:
	rm -f gnostic-swift-generator
	ln -s ../../gnostic-swift-generator 
	gnostic bookstore.json --swift-generator-out=Sources/Bookstore
	swift build

test:
	swift test
