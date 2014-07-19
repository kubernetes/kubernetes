pause: pause.asm
ifneq ($(shell uname), Linux)
	echo "Must build on Linux"
	exit 1
else
	nasm -o $@ $<
	chmod +x pause
endif

all: pause

clean:
	rm -f pause
