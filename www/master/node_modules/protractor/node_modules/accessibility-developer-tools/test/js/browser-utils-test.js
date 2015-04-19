module("matchSelector", {
  setup: function () {
    this.fixture_ = document.getElementById('qunit-fixture');
    this.matching_selector_ = '#qunit-fixture';
    this.mismatching_selector_ = '#not-fixture';
  }
});
test("nodes are the same", function () {
  equal(axs.browserUtils.matchSelector(this.fixture_, this.matching_selector_), true);
});

test("nodes are different", function () {
  equal(axs.browserUtils.matchSelector(this.fixture_, this.mismatching_selector_), false);
});
