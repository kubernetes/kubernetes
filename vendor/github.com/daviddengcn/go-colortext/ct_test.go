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
	}
	fmt.Println()
	ChangeColor(Red, true, White, false)
	fmt.Println("Before reset.")
	ChangeColor(Red, false, White, true)
	fmt.Println("Before reset.")
	ResetColor()
	fmt.Println("After reset.")
	fmt.Println("After reset.")
}

func TestForeground(t *testing.T) {
	ResetColor()
	defer ResetColor()

	fmt.Println("Please check the words under the following text shows with the corresponding front color:")

	colorToText := [...]string{
		Black:   "black",
		Red:     "red",
		Green:   "green",
		Yellow:  "yellow",
		Blue:    "blue",
		Magenta: "magenta",
		Cyan:    "cyan",
		White:   "white",
	}
	for i, txt := range colorToText {
		cl := Color(i)
		if cl != None {
			Foreground(cl, false)
			fmt.Print(txt, ",")
			Foreground(cl, true)
			fmt.Print(txt, ",")
		}
	}
	fmt.Println()
}

func TestBackground(t *testing.T) {
	ResetColor()
	defer ResetColor()

	fmt.Println("Please check the words under the following text shows with the corresponding background color:")

	colorToText := [...]string{
		Black:   "black",
		Red:     "red",
		Green:   "green",
		Yellow:  "yellow",
		Blue:    "blue",
		Magenta: "magenta",
		Cyan:    "cyan",
		White:   "white",
	}
	for i, txt := range colorToText {
		cl := Color(i)
		if cl != None {
			Background(cl, false)
			fmt.Print(txt, ",")
			Background(cl, true)
			fmt.Print(txt, ",")
		}
	}
	fmt.Println()
}
