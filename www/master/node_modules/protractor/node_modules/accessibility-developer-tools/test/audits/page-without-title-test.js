module("Page titles");

test("Page titles must be present and non-empty", function() {
    var rule = axs.AuditRules.getRule('pageWithoutTitle');

    // Remove the title element from the qunit test page.
    var title = document.querySelector('title');
    if (title && title.parentNode)
        title.parentNode.removeChild(title);

    // This one fails because there is no title element.
    equal(rule.run().result,
          axs.constants.AuditResult.FAIL);

    var head = document.querySelector('head');
    var blankTitle = document.createElement('title');
    head.appendChild(blankTitle);

    // This one fails because the title element is blank.
    equal(rule.run().result,
          axs.constants.AuditResult.FAIL);

    blankTitle.textContent = 'foo';
    equal(rule.run().result,
          axs.constants.AuditResult.PASS);

});