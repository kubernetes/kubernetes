package main

import (
	"fmt"
	"github.com/blang/semver"
)

func main() {
	v, err := semver.Parse("0.0.1-alpha.preview.222+123.github")
	if err != nil {
		fmt.Printf("Error while parsing (not valid): %q", err)
	}
	fmt.Printf("Version to string: %q\n", v)

	fmt.Printf("Major: %d\n", v.Major)
	fmt.Printf("Minor: %d\n", v.Minor)
	fmt.Printf("Patch: %d\n", v.Patch)

	// Prerelease versions
	if len(v.Pre) > 0 {
		fmt.Println("Prerelease versions:")
		for i, pre := range v.Pre {
			fmt.Printf("%d: %q\n", i, pre)
		}
	}

	// Build meta data
	if len(v.Build) > 0 {
		fmt.Println("Build meta data:")
		for i, build := range v.Build {
			fmt.Printf("%d: %q\n", i, build)
		}
	}

	// Make == Parse (Value), New for Pointer
	v001, err := semver.Make("0.0.1")

	fmt.Println("\nUse Version.Compare for comparisons (-1, 0, 1):")
	fmt.Printf("%q is greater than %q: Compare == %d\n", v001, v, v001.Compare(v))
	fmt.Printf("%q is less than %q: Compare == %d\n", v, v001, v.Compare(v001))
	fmt.Printf("%q is equal to %q: Compare == %d\n", v, v, v.Compare(v))

	fmt.Println("\nUse comparison helpers returning booleans:")
	fmt.Printf("%q is greater than %q: %t\n", v001, v, v001.GT(v))
	fmt.Printf("%q is greater than equal %q: %t\n", v001, v, v001.GTE(v))
	fmt.Printf("%q is greater than equal %q: %t\n", v, v, v.GTE(v))
	fmt.Printf("%q is less than %q: %t\n", v, v001, v.LT(v001))
	fmt.Printf("%q is less than equal %q: %t\n", v, v001, v.LTE(v001))
	fmt.Printf("%q is less than equal %q: %t\n", v, v, v.LTE(v))

	fmt.Println("\nManipulate Version in place:")
	v.Pre[0], err = semver.NewPRVersion("beta")
	if err != nil {
		fmt.Printf("Error parsing pre release version: %q", err)
	}
	fmt.Printf("Version to string: %q\n", v)

	fmt.Println("\nCompare Prerelease versions:")
	pre1, _ := semver.NewPRVersion("123")
	pre2, _ := semver.NewPRVersion("alpha")
	pre3, _ := semver.NewPRVersion("124")
	fmt.Printf("%q is less than %q: Compare == %d\n", pre1, pre2, pre1.Compare(pre2))
	fmt.Printf("%q is greater than %q: Compare == %d\n", pre3, pre1, pre3.Compare(pre1))
	fmt.Printf("%q is equal to %q: Compare == %d\n", pre1, pre1, pre1.Compare(pre1))

	fmt.Println("\nValidate versions:")
	v.Build[0] = "?"

	err = v.Validate()
	if err != nil {
		fmt.Printf("Validation failed: %s\n", err)
	}

	fmt.Println("Create valid build meta data:")
	b1, _ := semver.NewBuildVersion("build123")
	v.Build[0] = b1
	fmt.Printf("Version with new build version %q\n", v)

	_, err = semver.NewBuildVersion("build?123")
	if err != nil {
		fmt.Printf("Create build version failed: %s\n", err)
	}
}
