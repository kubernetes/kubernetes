module('NonExistentAriaRelatedElement');
[
  'aria-activedescendant',  // strictly speaking sometests do not apply to this
  'aria-controls',
  'aria-describedby',
  'aria-flowto',
  'aria-labelledby',
  'aria-owns'].forEach(function(testProp) {
    test('Element exists, single ' + testProp + ' value', function() {
        var fixture = document.getElementById('qunit-fixture');
        var referentElement = document.createElement('div');
        referentElement.textContent = 'label';
        referentElement.id = 'theLabel';
        fixture.appendChild(referentElement);

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'theLabel');
        fixture.appendChild(refererElement);

        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        deepEqual(rule.run({ scope: fixture }),
                  { elements: [], result: axs.constants.AuditResult.PASS });
    });

    test('Element doesn\'t exist, single ' + testProp + ' value', function() {
        var fixture = document.getElementById('qunit-fixture');

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'notALabel');
        fixture.appendChild(refererElement);
        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [refererElement]);
    });

    test('Element doesn\'t exist, single ' + testProp + ' value with aria-busy', function() {
        var fixture = document.getElementById('qunit-fixture');

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'notALabel');
        refererElement.setAttribute('aria-busy', 'true');
        fixture.appendChild(refererElement);
        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [refererElement]);
    });

    test('Element doesn\'t exist, single ' + testProp + ' value with aria-hidden', function() {
        var fixture = document.getElementById('qunit-fixture');

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'notALabel');
        refererElement.setAttribute('aria-hidden', 'true');
        fixture.appendChild(refererElement);
        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [refererElement]);
    });

    test('Multiple referent elements exist with ' + testProp, function() {
        var fixture = document.getElementById('qunit-fixture');
        var referentElement = document.createElement('div');
        referentElement.textContent = 'label';
        referentElement.id = 'theLabel';
        fixture.appendChild(referentElement);

        var referentElement2 = document.createElement('div');
        referentElement2.textContent = 'label2';
        referentElement2.id = 'theOtherLabel';
        fixture.appendChild(referentElement2);

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'theLabel theOtherLabel');
        fixture.appendChild(refererElement);

        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        deepEqual(rule.run({ scope: fixture }),
                  { elements: [], result: axs.constants.AuditResult.PASS });

    });

    test('One element doesn\'t exist, multiple ' + testProp, function() {
        var fixture = document.getElementById('qunit-fixture');

        var referentElement = document.createElement('div');
        referentElement.textContent = 'label';
        referentElement.id = 'theLabel';
        fixture.appendChild(referentElement);

        var refererElement = document.createElement('div');
        refererElement.setAttribute(testProp, 'theLabel notALabel');
        fixture.appendChild(refererElement);
        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        var result = rule.run({ scope: fixture });
        equal(result.result, axs.constants.AuditResult.FAIL);
        deepEqual(result.elements, [refererElement]);
    });

    test('Using ignoreSelectors with ' + testProp, function() {
        var fixture = document.getElementById('qunit-fixture');

        var referentElement = document.createElement('div');
        referentElement.textContent = 'label2';
        referentElement.id = 'theLabel2';
        fixture.appendChild(referentElement);

        var refererElement = document.createElement('div');
        refererElement.id = 'labelledbyElement2';
        refererElement.setAttribute(testProp, 'theLabel2 notALabel2');
        fixture.appendChild(refererElement);

        var rule = axs.AuditRules.getRule('nonExistentAriaRelatedElement');
        var ignoreSelectors = ['#labelledbyElement2'];
        var result = rule.run({ ignoreSelectors: ignoreSelectors, scope: fixture });
        equal(result.result, axs.constants.AuditResult.PASS);
    });
});
