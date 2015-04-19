module('BadAriaAttributeValue');

test('Empty idref value is ok', function() {
    var fixture = document.getElementById('qunit-fixture');
    var div = document.createElement('div');
    fixture.appendChild(div);
    div.setAttribute('aria-activedescendant', '');
    deepEqual(
      axs.AuditRules.getRule('badAriaAttributeValue').run({ scope: fixture }),
      { elements: [], result: axs.constants.AuditResult.PASS }
    );
});

test('Bad number value doesn\'t cause crash', function() {
    var fixture = document.getElementById('qunit-fixture');
    var div = document.createElement('div');
    fixture.appendChild(div);
    div.setAttribute('aria-valuenow', 'foo');
    deepEqual(
      axs.AuditRules.getRule('badAriaAttributeValue').run({ scope: fixture }),
      { elements: [div], result: axs.constants.AuditResult.FAIL }
    );
});

test('Good number value is good', function() {
    var fixture = document.getElementById('qunit-fixture');
    var div = document.createElement('div');
    fixture.appendChild(div);
    div.setAttribute('aria-valuenow', '10');
    deepEqual(
      axs.AuditRules.getRule('badAriaAttributeValue').run({ scope: fixture }),
      { elements: [], result: axs.constants.AuditResult.PASS }
    );
});
