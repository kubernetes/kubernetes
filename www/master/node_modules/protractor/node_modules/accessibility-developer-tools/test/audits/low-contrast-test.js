module("LowContrast");

test("No text = no relevant elements", function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.style.backgroundColor = 'white';
  div.style.color = 'white';
  fixture.appendChild(div);
  deepEqual(
    axs.AuditRules.getRule('lowContrastElements').run({ scope: fixture }),
    { result: axs.constants.AuditResult.NA }
  );
});

test("Black on white = no problem", function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.style.backgroundColor = 'white';
  div.style.color = 'black';
  div.textContent = 'Some text';
  fixture.appendChild(div);
  deepEqual(
    axs.AuditRules.getRule('lowContrastElements').run({ scope: fixture }),
    { elements: [], result: axs.constants.AuditResult.PASS }
  );
});

test("Low contrast = fail", function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.style.backgroundColor = 'white';
  div.style.color = '#aaa';  // Contrast ratio = 2.32
  div.textContent = 'Some text';
  fixture.appendChild(div);
  deepEqual(
    axs.AuditRules.getRule('lowContrastElements').run({ scope: fixture }),
    { elements: [div], result: axs.constants.AuditResult.FAIL }
  );
});

test("Opacity is handled", function() {
  // Setup fixture
  var fixture = document.getElementById('qunit-fixture');
  var elementWithOpacity = document.createElement('div');
  elementWithOpacity.style.opacity = '0.4';
  elementWithOpacity.textContent = 'Some text';
  fixture.appendChild(elementWithOpacity);
  deepEqual(
    axs.AuditRules.getRule('lowContrastElements').run({ scope: fixture }),
    { elements: [elementWithOpacity], result: axs.constants.AuditResult.FAIL }
  );
});

test("Uses tolerance value", function() {
  var fixture = document.getElementById('qunit-fixture');
  var div = document.createElement('div');
  div.style.backgroundColor = 'white';
  div.style.color = '#777'; // Contrast ratio = 4.48
  div.textContent = 'Some text';
  fixture.appendChild(div);
  deepEqual(
    axs.AuditRules.getRule('lowContrastElements').run({ scope: fixture }),
    { elements: [], result: axs.constants.AuditResult.PASS }
  );
});
