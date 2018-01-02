.globl _start
.text
_start:
	xorl	%eax, %eax
	incl	%eax
	movb	$0, %bl
	int	$0x80
