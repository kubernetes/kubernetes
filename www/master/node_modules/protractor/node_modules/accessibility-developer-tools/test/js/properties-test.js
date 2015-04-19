module("Text Descendant", {
});

test("Find text descendants in an iframe.", function() {
    // Setup fixture
    var fixture = document.getElementById('qunit-fixture');

    var iframe = document.createElement('iframe');
    var html = '<body><div id="foo">bar</div></body>';
    fixture.appendChild(iframe);
    iframe.contentWindow.document.open();
    iframe.contentWindow.document.write(html);
    iframe.contentWindow.document.close();

    var foo = iframe.contentDocument.getElementById("foo");

    equal(axs.properties.hasDirectTextDescendant(foo), true);
});

module('findTextAlternatives', {
    setup: function () {
        this.fixture_ = document.getElementById('qunit-fixture');
    }
});
test('returns the calculated text alternative for the given element', function() {
    var targetNode = document.createElement('select');
    this.fixture_.appendChild(targetNode);

    try {
        equal(axs.properties.findTextAlternatives(targetNode, {}, true), '');
    } catch(e) {
        ok(false, 'Threw exception');
    }
});

module('getTextFromHostLanguageAttributes', {
    setup: function () {
        this.fixture_ = document.getElementById('qunit-fixture');
    }
});
test('does not crash when targetNode has a numeric id attribute', function() {
    var targetNode = document.createElement('input');
    targetNode.setAttribute('id', '123_user');
    this.fixture_.appendChild(targetNode);

    try {
        equal(axs.properties.getTextFromHostLanguageAttributes(targetNode, {}, null), null);
    } catch(e) {
        ok(false, 'Threw exception: ' + e);
    }
});

module('getFocusProperties', {});
test('Get focus properties', function() {
    // Setup fixture
    var fixture = document.getElementById('qunit-fixture');
    fixture.style.top = 0;
    fixture.style.left = 0;

    var html = '<div id="overlapped" tabindex="0">Overlapped element</div>' +
               '<div id="overlapping" style="font-size: 48px; ' +
               'position: relative; top: -40px; height: 40px; ' +
               'background: rgba(255, 255, 255, 0.5);">Overlapping div</div>';
    fixture.innerHTML = html;

    var overlapped = document.getElementById('overlapped');
    var overlapping = document.getElementById('overlapping');

    var rect = overlapped.getBoundingClientRect();
    var center_x = (rect.left + rect.right) / 2;
    var center_y = (rect.top + rect.bottom) / 2;
    var elementAtPoint = document.elementFromPoint(center_x, center_y);

    var focusProperties = axs.properties.getFocusProperties(overlapped);
    if (elementAtPoint != null) {
        deepEqual(focusProperties,
                  { tabindex: { value: "0", valid: true },
                    visible: { value: false,
                               hidden: {
                                   value: false,
                                   valid: false
                               },
                               valid: false,
                               overlappingElements: [overlapping] } });
    } else {
        // This will occur if running in phantomjs.
        deepEqual(focusProperties,
                  { tabindex: { value: "0", valid: true },
                    visible: { value: true, valid: true } });
    }
});

module("getTextFromDescendantContent", {
  setup: function () {
    this.fixture_ = document.getElementById('qunit-fixture');
  }
});
test("returns text from the descendants of the element", function() {
  var html = '<label>\n' +
             '  <input type="radio" id="reason_Screenshot" name="reason" value="screenshot"></input>\n' +
             '</label>'
  this.fixture_.innerHTML = html;
  var targetNode = this.fixture_.querySelector('label');

  try {
    equal(axs.properties.getTextFromDescendantContent(targetNode), "");
    return ok(true);
  } catch( e ) {
    return ok(false, "Threw exception: " + e);
  }
});
