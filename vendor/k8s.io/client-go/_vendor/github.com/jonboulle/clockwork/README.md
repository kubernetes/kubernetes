clockwork
=========

a simple fake clock for golang

# Usage

Replace uses of the `time` package with the `clockwork.Clock` interface instead.

For example, instead of using `time.Sleep` directly:

```
func my_func() {
	time.Sleep(3 * time.Second)
	do_something()
}
```

inject a clock and use its `Sleep` method instead:

```
func my_func(clock clockwork.Clock) {
	clock.Sleep(3 * time.Second)
	do_something()
}
```

Now you can easily test `my_func` with a `FakeClock`:

```
func TestMyFunc(t *testing.T) {
	c := clockwork.NewFakeClock()

	// Start our sleepy function
	my_func(c)

	// Ensure we wait until my_func is sleeping
	c.BlockUntil(1)

	assert_state()

	// Advance the FakeClock forward in time
	c.Advance(3)

	assert_state()
}
```

and in production builds, simply inject the real clock instead:
```
my_func(clockwork.NewRealClock())
```

See [example_test.go](example_test.go) for a full example.

# Credits

Inspired by @wickman's [threaded fake clock](https://gist.github.com/wickman/3840816), and the [Golang playground](http://blog.golang.org/playground#Faking time)
