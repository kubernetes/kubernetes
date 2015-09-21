package ct

import (
	"fmt"
	"testing"
)

func TestChangeColor(t *testing.T) {
	defer ResetColor()
	fmt.Println("Normal text...")
	text := "This is an demo of using ChangeColor to output colorful texts"
	i := 1
	for _, c := range text {
		ChangeColor(Color(i/2%8)+Black, i%2 == 1, Color((i+2)/2%8)+Black, false)
		fmt.Print(string(c))
		i++
	} // for c
	fmt.Println()
	ChangeColor(Red, true, White, false)
	fmt.Println("Before reset.")
	ChangeColor(Red, false, White, true)
	fmt.Println("Before reset.")
	ResetColor()
	fmt.Println("After reset.")
	fmt.Println("After reset.")
}
