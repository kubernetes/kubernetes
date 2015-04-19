(function() {  // scope to avoid leaking helpers and variables to global namespace
    var RULE_NAME = 'ariaOwnsDescendant';
    // TODO(RickSBrown): Refactor tests, there is too much cut and paste reuse;
    module('AriaOwnsDescendant');

    test('Element owns a sibling', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owned = fixture.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        var owner = fixture.appendChild(document.createElement('div'));
        owner.setAttribute('aria-owns', owned.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        deepEqual(rule.run({ scope: fixture }),
                  { elements: [], result: axs.constants.AuditResult.PASS });
    });

    test('Element owns multiple siblings', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owned = fixture.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        var owned2 = fixture.appendChild(document.createElement('div'));
        owned2.id = 'ownedElement2';
        var owner = fixture.appendChild(document.createElement('div'));
        owner.setAttribute('aria-owns', owned.id + ' ' + owned2.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        deepEqual(rule.run({ scope: fixture }),
                  { elements: [], result: axs.constants.AuditResult.PASS });
    });

    test('Element owns a descendant', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owner = fixture.appendChild(document.createElement('div'));
        var owned = owner.appendChild(document.createElement('div'));
        for (var i = 0; i < 9; i++)  // ensure it works on descendants, not just children
            owned = owned.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        owner.setAttribute('aria-owns', owned.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [owner]);
    });

    test('Element owns multiple descendants', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owner = fixture.appendChild(document.createElement('div'));
        var owned = owner.appendChild(document.createElement('div'));
        for (var i = 0; i < 9; i++)  // ensure it works on descendants, not just children
            owned = owned.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        var owned2 = owner.appendChild(document.createElement('div'));
        owned2.id = 'ownedElement2';
        owner.setAttribute('aria-owns', owned.id + ' ' + owned2.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [owner]);
    });

    test('Element owns one sibling one descendant', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owner = fixture.appendChild(document.createElement('div'));
        var owned = owner.appendChild(document.createElement('div'));
        for (var i = 0; i < 9; i++)  // ensure it works on descendants, not just children
            owned = owned.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        var owned2 = fixture.appendChild(document.createElement('div'));
        owned2.id = 'ownedElement2';
        owner.setAttribute('aria-owns', owned.id + ' ' + owned2.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [owner]);
    });

    test('Using ignoreSelectors - element owns a descendant', function() {
        var fixture = document.getElementById('qunit-fixture');
        var owner = fixture.appendChild(document.createElement('div'));
        var owned = owner.appendChild(document.createElement('div'));
        for (var i = 0; i < 9; i++)  // ensure it works on descendants, not just children
            owned = owned.appendChild(document.createElement('div'));
        owned.id = 'ownedElement';
        owner.setAttribute('aria-owns', owned.id);
        var rule = axs.AuditRules.getRule(RULE_NAME);
        var ignoreSelectors = ['#' + (owner.id = 'ownerElement')];
        var result = rule.run({
            ignoreSelectors: ignoreSelectors,
            scope: fixture });
        equal(result.result, axs.constants.AuditResult.PASS);
        deepEqual(result.elements, []);
    });
})();
