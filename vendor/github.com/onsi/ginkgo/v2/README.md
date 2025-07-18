![Ginkgo](https://onsi.github.io/ginkgo/images/ginkgo.png)

[![test](https://github.com/onsi/ginkgo/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/onsi/ginkgo/actions?query=workflow%3Atest+branch%3Amaster) | [Ginkgo Docs](https://onsi.github.io/ginkgo/)

---

# Ginkgo

Ginkgo is a mature testing framework for Go designed to help you write expressive specs.  Ginkgo builds on top of Go's `testing` foundation and is complemented by the [Gomega](https://github.com/onsi/gomega) matcher library.  Together, Ginkgo and Gomega let you express the intent behind your specs clearly:

```go
import (
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    ...
)

var _ = Describe("Checking books out of the library", Label("library"), func() {
    var library *libraries.Library
    var book *books.Book
    var valjean *users.User
    BeforeEach(func() {
        library = libraries.NewClient()
        book = &books.Book{
            Title: "Les Miserables",
            Author: "Victor Hugo",
        }
        valjean = users.NewUser("Jean Valjean")
    })

    When("the library has the book in question", func() {
        BeforeEach(func(ctx SpecContext) {
            Expect(library.Store(ctx, book)).To(Succeed())
        })

        Context("and the book is available", func() {
            It("lends it to the reader", func(ctx SpecContext) {
                Expect(valjean.Checkout(ctx, library, "Les Miserables")).To(Succeed())
                Expect(valjean.Books()).To(ContainElement(book))
                Expect(library.UserWithBook(ctx, book)).To(Equal(valjean))
            }, SpecTimeout(time.Second * 5))
        })

        Context("but the book has already been checked out", func() {
            var javert *users.User
            BeforeEach(func(ctx SpecContext) {
                javert = users.NewUser("Javert")
                Expect(javert.Checkout(ctx, library, "Les Miserables")).To(Succeed())
            })

            It("tells the user", func(ctx SpecContext) {
                err := valjean.Checkout(ctx, library, "Les Miserables")
                Expect(err).To(MatchError("Les Miserables is currently checked out"))
            }, SpecTimeout(time.Second * 5))

            It("lets the user place a hold and get notified later", func(ctx SpecContext) {
                Expect(valjean.Hold(ctx, library, "Les Miserables")).To(Succeed())
                Expect(valjean.Holds(ctx)).To(ContainElement(book))

                By("when Javert returns the book")
                Expect(javert.Return(ctx, library, book)).To(Succeed())

                By("it eventually informs Valjean")
                notification := "Les Miserables is ready for pick up"
                Eventually(ctx, valjean.Notifications).Should(ContainElement(notification))

                Expect(valjean.Checkout(ctx, library, "Les Miserables")).To(Succeed())
                Expect(valjean.Books(ctx)).To(ContainElement(book))
                Expect(valjean.Holds(ctx)).To(BeEmpty())
            }, SpecTimeout(time.Second * 10))
        })  
    })

    When("the library does not have the book in question", func() {
        It("tells the reader the book is unavailable", func(ctx SpecContext) {
            err := valjean.Checkout(ctx, library, "Les Miserables")
            Expect(err).To(MatchError("Les Miserables is not in the library catalog"))
        }, SpecTimeout(time.Second * 5))
    })
})
```

Jump to the [docs](https://onsi.github.io/ginkgo/) to learn more.  It's easy to [bootstrap](https://onsi.github.io/ginkgo/#bootstrapping-a-suite) and start writing your [first specs](https://onsi.github.io/ginkgo/#adding-specs-to-a-suite).

If you have a question, comment, bug report, feature request, etc. please open a [GitHub issue](https://github.com/onsi/ginkgo/issues/new), or visit the [Ginkgo Slack channel](https://app.slack.com/client/T029RQSE6/CQQ50BBNW).

## Capabilities

Whether writing basic unit specs, complex integration specs, or even performance specs - Ginkgo gives you an expressive Domain-Specific Language (DSL) that will be familiar to users coming from frameworks such as [Quick](https://github.com/Quick/Quick), [RSpec](https://rspec.info), [Jasmine](https://jasmine.github.io), and [Busted](https://lunarmodules.github.io/busted/).  This style of testing is sometimes referred to as "Behavior-Driven Development" (BDD) though Ginkgo's utility extends beyond acceptance-level testing.

With Ginkgo's DSL you can use nestable [`Describe`, `Context` and `When` container nodes](https://onsi.github.io/ginkgo/#organizing-specs-with-container-nodes) to help you organize your specs.  [`BeforeEach` and `AfterEach` setup nodes](https://onsi.github.io/ginkgo/#extracting-common-setup-beforeeach) for setup and cleanup.  [`It` and `Specify` subject nodes](https://onsi.github.io/ginkgo/#spec-subjects-it) that hold your assertions. [`BeforeSuite` and `AfterSuite` nodes](https://onsi.github.io/ginkgo/#suite-setup-and-cleanup-beforesuite-and-aftersuite) to prep for and cleanup after a suite... and [much more!](https://onsi.github.io/ginkgo/#writing-specs).

At runtime, Ginkgo can run your specs in reproducibly [random order](https://onsi.github.io/ginkgo/#spec-randomization) and has sophisticated support for [spec parallelization](https://onsi.github.io/ginkgo/#spec-parallelization).  In fact, running specs in parallel is as easy as

```bash
ginkgo -p
```

By following [established patterns for writing parallel specs](https://onsi.github.io/ginkgo/#patterns-for-parallel-integration-specs) you can build even large, complex integration suites that parallelize cleanly and run performantly.  And you don't have to worry about your spec suite hanging or leaving a mess behind - Ginkgo provides a per-node `context.Context` and the capability to interrupt the spec after a set period of time - and then clean up.

As your suites grow Ginkgo helps you keep your specs organized with [labels](https://onsi.github.io/ginkgo/#spec-labels) and lets you easily run [subsets of specs](https://onsi.github.io/ginkgo/#filtering-specs), either [programmatically](https://onsi.github.io/ginkgo/#focused-specs) or on the [command line](https://onsi.github.io/ginkgo/#combining-filters).  And Ginkgo's reporting infrastructure generates machine-readable output in a [variety of formats](https://onsi.github.io/ginkgo/#generating-machine-readable-reports) _and_ allows you to build your own [custom reporting infrastructure](https://onsi.github.io/ginkgo/#generating-reports-programmatically).

Ginkgo ships with `ginkgo`, a [command line tool](https://onsi.github.io/ginkgo/#ginkgo-cli-overview) with support for generating, running, filtering, and profiling Ginkgo suites.  You can even have Ginkgo automatically run your specs when it detects a change with `ginkgo watch`, enabling rapid feedback loops during test-driven development.

And that's just Ginkgo!  [Gomega](https://onsi.github.io/gomega/) brings a rich, mature, family of [assertions and matchers](https://onsi.github.io/gomega/#provided-matchers) to your suites.  With Gomega you can easily mix [synchronous and asynchronous assertions](https://onsi.github.io/ginkgo/#patterns-for-asynchronous-testing) in your specs.  You can even build your own set of expressive domain-specific matchers quickly and easily by composing Gomega's [existing building blocks](https://onsi.github.io/ginkgo/#building-custom-matchers).

Happy Testing!

## License

Ginkgo is MIT-Licensed

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
