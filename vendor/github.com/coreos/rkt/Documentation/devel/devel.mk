%.png: %.neato
	neato -Tpng -Gdpi=75 $< -o $@

%.png: %.dot
	dot -Tpng -Gdpi=75 $< -o $@

images: \
	Documentation/devel/image-logical-blocks.png \
	Documentation/devel/image-chain.png \
	Documentation/devel/execution-flow.png \
	Documentation/devel/execution-flow-fly.png \
	Documentation/devel/execution-flow-systemd.png \
	Documentation/devel/mutable.png \
	Documentation/devel/mutable-app.png
