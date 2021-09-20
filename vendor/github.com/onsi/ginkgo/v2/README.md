![Ginkgo](https://onsi.github.io/ginkgo/images/ginkgo.png)

[![test](https://github.com/onsi/ginkgo/workflows/test/badge.svg?branch=master)](https://github.com/onsi/ginkgo/actions?query=workflow%3Atest+branch%3Amaster)

Ginkgo is a mature, well-established, testing framework for Go designed to help you write expressive specs.  Ginkgo builds on top of Go's `testing` foundation and is complemented by the [Gomega](https://github.com/onsi/gomega) matcher library.  Together, Ginkgo and Gomega let you express the intent behind your specs clearly:

```go
import (
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    ...
)

Describe("Checking books out of the library", func() {
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
        BeforeEach(func() {
            Expect(library.Store(book)).To(Succeed())
        })

        Context("and the book is available", func() {
            It("lends it to the reader", func() {
                Expect(valjean.Checkout(library, "Les Miserables")).To(Succeed())
                Expect(valjean.Books()).To(ContainElement(book))
                Expect(library.UserWithBook(book)).To(Equal(valjean))
            })
        })

        Context("but the book has already been checked out", func() {
            var javert *users.User
            BeforeEach(func() {
                javert = users.NewUser("Javert")
                Expect(javert.Checkout(library, "Les Miserables")).To(Succeed())
            })

            It("tells the user", func() {
                err := valjean.Checkout(library, "Les Miserables")
                Expect(error).To(MatchError("Les Miserables is currently checked out"))
            })

            It("lets the user place a hold and get notified later", func() {
                Expect(valjean.Hold(library, "Les Miserables")).To(Succeed())
                Expect(valjean.Holds()).To(ContainElement(book))

                By("when Javert returns the book")
                Expect(javert.Return(library, book)).To(Succeed())

                By("it eventually informs Valjean")
                notification := "Les Miserables is ready for pick up"
                Eventually(valjean.Notifications).Should(ContainElement(notification))

                Expect(valjean.Checkout(library, "Les Miserables")).To(Succeed())
                Expect(valjean.Books()).To(ContainElement(book))
                Expect(valjean.Holds()).To(BeEmpty())
            })
        })  
    })

    When("the library does not have the book in question", func() {
        It("tells the reader the book is unavailable", func() {
            err := valjean.Checkout(library, "Les Miserables")
            Expect(error).To(MatchError("Les Miserables is not in the library catalog"))
        })
    })
})
```

Jump to the [docs](https://onsi.github.io/ginkgo/) to learn more!  It's easy to [bootstrap](https://onsi.github.io/ginkgo/#bootstrapping-a-suite) and start writing your [first specs](https://onsi.github.io/ginkgo/#adding-specs-to-a-suite).

If you have a question, comment, bug report, feature request, etc. please open a GitHub issue, or visit the [Ginkgo Slack channel](https://app.slack.com/client/T029RQSE6/CQQ50BBNW).

## Capabilities

Whether writing basic unit specs, complex integration specs, or even performance specs - Ginkgo gives you an expressive Domain-Specific Language (DSL) that will be familiar to users coming from frameworks such as [Quick](https://github.com/Quick/Quick), [RSpec](https://rspec.info), [Jasmine](https://jasmine.github.io), [Busted](https://olivinelabs.com/busted/).  This style of testing is sometimes referred to as "Behavior-Driven Development" (BDD) though Ginkgo's utility extends beyond acceptance-level testing.

With Ginkgo's DSL you can use nestable [`Describe`, `Context` and `When` container nodes](https://onsi.github.io/ginkgo/#organizing-specs-with-container-nodes) to help you organize your specs.  [`BeforeEach` and `AfterEach` setup nodes](https://onsi.github.io/ginkgo/#extracting-common-setup-beforeeach) for setup and cleanup.  [`It` and `Specify` subject nodes](https://onsi.github.io/ginkgo/#spec-subjects-it) that hold your assertions. [`BeforeSuite` and `AfterSuite` nodes](https://onsi.github.io/ginkgo/#suite-setup-and-cleanup-beforesuite-and-aftersuite) to prep for and cleanup after a suite... and [much more!](https://onsi.github.io/ginkgo/#writing-specs)

At runtime, Ginkgo can run your specs in reproducibly [random order](https://onsi.github.io/ginkgo/#spec-randomization) and has sophisticated support for [spec parallelization](https://onsi.github.io/ginkgo/#spec-parallelization).  In fact, running specs in parallel is as easy as

```bash
ginkgo -p
```

By following [established patterns for writing parallel specs](https://onsi.github.io/ginkgo/#patterns-for-parallel-integration-specs) you can build even large, complex integration suites that parallelize cleanly and run performantly.

As your suites grow Ginkgo helps you keep your specs organized with [labels](https://onsi.github.io/ginkgo/#spec-labels) and lets you easily run [subsets of specs](https://onsi.github.io/ginkgo/#filtering-specs), either [programatically](https://onsi.github.io/ginkgo/#focused-specs) or on the [command line](https://onsi.github.io/ginkgo/#combining-filters).  And Ginkgo's reporting infrastructure will generate machine-readable output in a [variety of formats](https://onsi.github.io/ginkgo/#generating-machine-readable-reports) _and_ allow you to build your own [custom reporting infrastructure](https://onsi.github.io/ginkgo/#generating-reports-programmatically).

Ginkgo ships with `ginkgo`, a [command line tool](https://onsi.github.io/ginkgo/#ginkgo-cli-overview) with powerful support for generating, running, filtering, and profiling Ginkgo suites.  You can even have Ginkgo automatically run your specs when it detects a change with `ginkgo watch`, enabling rapid feedback loops during test-driven development.

And that's just Ginkgo!  [Gomega](https://onsi.github.io/gomega/) brings a rich, mature, family of [assertions and matchers](https://onsi.github.io/gomega/#provided-matchers) to your suites.  With Gomega you can easily mix [synchronous and asynchronous assertions](https://onsi.github.io/ginkgo/#patterns-for-asynchronous-testing) in your specs.  You can even build your own set of expressive domain-specific matchers quickly and easily by composing Gomega's [existing building blocks](https://onsi.github.io/ginkgo/#building-custom-matchers).

Happy Testing!

## License

Ginkgo is MIT-Licensed

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
