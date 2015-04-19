codemirror30 = "// All functions that need access to the editor's state live inside\n\
// the CodeMirror function. Below that, at the bottom of the file,\n\
// some utilities are defined.\n\
\n\
// CodeMirror is the only global var we claim\n\
window.CodeMirror = (function() {\n\
  \"use strict\";\n\
\n\
  // BROWSER SNIFFING\n\
\n\
  // Crude, but necessary to handle a number of hard-to-feature-detect\n\
  // bugs and behavior differences.\n\
  var gecko = /gecko\\/\\d{7}/i.test(navigator.userAgent);\n\
  var ie = /MSIE \\d/.test(navigator.userAgent);\n\
  var ie_lt8 = /MSIE [1-7]\\b/.test(navigator.userAgent);\n\
  var ie_lt9 = /MSIE [1-8]\\b/.test(navigator.userAgent);\n\
  var webkit = /WebKit\\//.test(navigator.userAgent);\n\
  var chrome = /Chrome\\//.test(navigator.userAgent);\n\
  var opera = /Opera\\//.test(navigator.userAgent);\n\
  var safari = /Apple Computer/.test(navigator.vendor);\n\
  var khtml = /KHTML\\//.test(navigator.userAgent);\n\
  var mac_geLion = /Mac OS X 10\\D([7-9]|\\d\\d)\\D/.test(navigator.userAgent);\n\
\n\
  var ios = /AppleWebKit/.test(navigator.userAgent) && /Mobile\\/\\w+/.test(navigator.userAgent);\n\
  var mac = ios || /Mac/.test(navigator.platform);\n\
  var win = /Win/.test(navigator.platform);\n\
\n\
  // CONSTRUCTOR\n\
\n\
  function CodeMirror(place, options) {\n\
    if (!(this instanceof CodeMirror)) return new CodeMirror(place, options, true);\n\
    \n\
    this.options = options = options || {};\n\
    // Determine effective options based on given values and defaults.\n\
    for (var opt in defaults) if (!options.hasOwnProperty(opt) && defaults.hasOwnProperty(opt))\n\
      options[opt] = defaults[opt];\n\
    setGuttersForLineNumbers(options);\n\
\n\
    var display = this.display = makeDisplay(place);\n\
    display.wrapper.CodeMirror = this;\n\
    updateGutters(this);\n\
    themeChanged(this);\n\
    keyMapChanged(this);\n\
    if (options.tabindex != null) display.input.tabIndex = options.tabindex;\n\
    if (options.autofocus) focusInput(this);\n\
    if (options.lineWrapping) display.wrapper.className += \" CodeMirror-wrap\";\n\
\n\
    var doc = new BranchChunk([new LeafChunk([new Line(\"\", null, textHeight(display))])]);\n\
    // frontier is the point up to which the content has been parsed,\n\
    doc.frontier = 0;\n\
    doc.highlight = new Delayed();\n\
    doc.tabSize = options.tabSize;\n\
    // The selection. These are always maintained to point at valid\n\
    // positions. Inverted is used to remember that the user is\n\
    // selecting bottom-to-top.\n\
    this.view = {\n\
      doc: doc,\n\
      sel: {from: {line: 0, ch: 0}, to: {line: 0, ch: 0}, inverted: false, shift: false},\n\
      scrollTop: 0, scrollLeft: 0,\n\
      overwrite: false, focused: false,\n\
      // Tracks the maximum line length so that\n\
      // the horizontal scrollbar can be kept\n\
      // static when scrolling.\n\
      maxLine: getLine(doc, 0),\n\
      maxLineChanged: false,\n\
      suppressEdits: false,\n\
      goalColumn: null\n\
    };\n\
    loadMode(this);\n\
\n\
    // Initialize the content.\n\
    this.setValue(options.value || \"\");\n\
    doc.history = new History();\n\
\n\
    registerEventHandlers(this);\n\
    // IE throws unspecified error in certain cases, when\n\
    // trying to access activeElement before onload\n\
    var hasFocus; try { hasFocus = (document.activeElement == display.input); } catch(e) { }\n\
    if (hasFocus || options.autofocus) setTimeout(bind(onFocus, this), 20);\n\
    else onBlur(this);\n\
\n\
    for (var opt in optionHandlers)\n\
      if (optionHandlers.propertyIsEnumerable(opt))\n\
        optionHandlers[opt](this, options[opt]);\n\
  }\n\
\n\
  // DISPLAY CONSTRUCTOR\n\
\n\
  function makeDisplay(place) {\n\
    var d = {};\n\
    var input = d.input = elt(\"textarea\", null, null, \"position: absolute; padding: 0; width: 1px; height: 1em; outline: none;\");\n\
    input.setAttribute(\"wrap\", \"off\"); input.setAttribute(\"autocorrect\", \"off\"); input.setAttribute(\"autocapitalize\", \"off\");\n\
    // Wraps and hides input textarea\n\
    d.inputDiv = elt(\"div\", [input], null, \"overflow: hidden; position: relative; width: 3px; height: 0px;\");\n\
    // The actual fake scrollbars.\n\
    d.scrollbarH = elt(\"div\", [elt(\"div\", null, null, \"height: 1px\")], \"CodeMirror-hscrollbar\");\n\
    d.scrollbarV = elt(\"div\", [elt(\"div\", null, null, \"width: 1px\")], \"CodeMirror-vscrollbar\");\n\
    d.scrollbarFiller = elt(\"div\", null, \"CodeMirror-scrollbar-filler\");\n\
    // DIVs containing the selection and the actual code\n\
    d.lineDiv = elt(\"div\");\n\
    d.selectionDiv = elt(\"div\", null, null, \"position: relative; z-index: 1\");\n\
    // Blinky cursor, and element used to ensure cursor fits at the end of a line\n\
    d.cursor = elt(\"pre\", \"\\u00a0\", \"CodeMirror-cursor\");\n\
    // Secondary cursor, shown when on a 'jump' in bi-directional text\n\
    d.otherCursor = elt(\"pre\", \"\\u00a0\", \"CodeMirror-cursor CodeMirror-secondarycursor\");\n\
    // Used to measure text size\n\
    d.measure = elt(\"div\", null, \"CodeMirror-measure\");\n\
    // Wraps everything that needs to exist inside the vertically-padded coordinate system\n\
    d.lineSpace = elt(\"div\", [d.measure, d.cursor, d.otherCursor, d.selectionDiv, d.lineDiv],\n\
                         null, \"position: relative; outline: none\");\n\
    // Moved around its parent to cover visible view\n\
    d.mover = elt(\"div\", [elt(\"div\", [d.lineSpace], \"CodeMirror-lines\")], null, \"position: relative\");\n\
    d.gutters = elt(\"div\", null, \"CodeMirror-gutters\");\n\
    d.lineGutter = null;\n\
    // Set to the height of the text, causes scrolling\n\
    d.sizer = elt(\"div\", [d.mover], \"CodeMirror-sizer\");\n\
    // D is needed because behavior of elts with overflow: auto and padding is inconsistent across browsers\n\
    d.heightForcer = elt(\"div\", \"\\u00a0\", null, \"position: absolute; height: \" + scrollerCutOff + \"px\");\n\
    // Provides scrolling\n\
    d.scroller = elt(\"div\", [d.sizer, d.heightForcer], \"CodeMirror-scroll\");\n\
    d.scroller.setAttribute(\"tabIndex\", \"-1\");\n\
    // The element in which the editor lives.\n\
    d.wrapper = elt(\"div\", [d.gutters, d.inputDiv, d.scrollbarH, d.scrollbarV,\n\
                            d.scrollbarFiller, d.scroller], \"CodeMirror\");\n\
    // Work around IE7 z-index bug\n\
    if (ie_lt8) { d.gutters.style.zIndex = -1; d.scroller.style.paddingRight = 0; }\n\
    if (place.appendChild) place.appendChild(d.wrapper); else place(d.wrapper);\n\
\n\
    // Needed to hide big blue blinking cursor on Mobile Safari\n\
    if (ios) input.style.width = \"0px\";\n\
    if (!webkit) d.scroller.draggable = true;\n\
    // Needed to handle Tab key in KHTML\n\
    if (khtml) { d.inputDiv.style.height = \"1px\"; d.inputDiv.style.position = \"absolute\"; }\n\
    // Need to set a minimum width to see the scrollbar on IE7 (but must not set it on IE8).\n\
    else if (ie_lt8) d.scrollbarH.style.minWidth = d.scrollbarV.style.minWidth = \"18px\";\n\
\n\
    // Current visible range (may be bigger than the view window).\n\
    d.viewOffset = d.showingFrom = d.showingTo = d.lastSizeC = 0;\n\
\n\
    // Used to only resize the line number gutter when necessary (when\n\
    // the amount of lines crosses a boundary that makes its width change)\n\
    d.lineNumWidth = d.lineNumChars = null;\n\
    // See readInput and resetInput\n\
    d.prevInput = \"\";\n\
    // Set to true when a non-horizontal-scrolling widget is added. As\n\
    // an optimization, widget aligning is skipped when d is false.\n\
    d.alignWidgets = false;\n\
    // Flag that indicates whether we currently expect input to appear\n\
    // (after some event like 'keypress' or 'input') and are polling\n\
    // intensively.\n\
    d.pollingFast = false;\n\
    // Self-resetting timeout for the poller\n\
    d.poll = new Delayed();\n\
    // True when a drag from the editor is active\n\
    d.draggingText = false;\n\
\n\
    d.cachedCharWidth = d.cachedTextHeight = null;\n\
    d.measureLineCache = [];\n\
    d.measureLineCache.pos = 0;\n\
\n\
    // Tracks when resetInput has punted to just putting a short\n\
    // string instead of the (large) selection.\n\
    d.inaccurateSelection = false;\n\
\n\
    return d;\n\
  }\n\
\n\
  // STATE UPDATES\n\
\n\
  // Used to get the editor into a consistent state again when options change.\n\
\n\
  function loadMode(cm) {\n\
    var doc = cm.view.doc;\n\
    doc.mode = CodeMirror.getMode(cm.options, cm.options.mode);\n\
    doc.iter(0, doc.size, function(line) { line.stateAfter = null; });\n\
    doc.frontier = 0;\n\
    startWorker(cm, 100);\n\
  }\n\
\n\
  function wrappingChanged(cm) {\n\
    var doc = cm.view.doc;\n\
    if (cm.options.lineWrapping) {\n\
      cm.display.wrapper.className += \" CodeMirror-wrap\";\n\
      var perLine = cm.display.wrapper.clientWidth / charWidth(cm.display) - 3;\n\
      doc.iter(0, doc.size, function(line) {\n\
        if (line.hidden) return;\n\
        var guess = Math.ceil(line.text.length / perLine) || 1;\n\
        if (guess != 1) updateLineHeight(line, guess);\n\
      });\n\
      cm.display.sizer.style.minWidth = \"\";\n\
    } else {\n\
      cm.display.wrapper.className = cm.display.wrapper.className.replace(\" CodeMirror-wrap\", \"\");\n\
      computeMaxLength(cm.view);\n\
      doc.iter(0, doc.size, function(line) {\n\
        if (line.height != 1 && !line.hidden) updateLineHeight(line, 1);\n\
      });\n\
    }\n\
    regChange(cm, 0, doc.size);\n\
  }\n\
\n\
  function keyMapChanged(cm) {\n\
    var style = keyMap[cm.options.keyMap].style;\n\
    cm.display.wrapper.className = cm.display.wrapper.className.replace(/\\s*cm-keymap-\\S+/g, \"\") +\n\
      (style ? \" cm-keymap-\" + style : \"\");\n\
  }\n\
\n\
  function themeChanged(cm) {\n\
    cm.display.wrapper.className = cm.display.wrapper.className.replace(/\\s*cm-s-\\S+/g, \"\") +\n\
      cm.options.theme.replace(/(^|\\s)\\s*/g, \" cm-s-\");\n\
  }\n\
\n\
  function guttersChanged(cm) {\n\
    updateGutters(cm);\n\
    updateDisplay(cm, true);\n\
  }\n\
\n\
  function updateGutters(cm) {\n\
    var gutters = cm.display.gutters, specs = cm.options.gutters;\n\
    removeChildren(gutters);\n\
    for (var i = 0; i < specs.length; ++i) {\n\
      var gutterClass = specs[i];\n\
      var gElt = gutters.appendChild(elt(\"div\", null, \"CodeMirror-gutter \" + gutterClass));\n\
      if (gutterClass == \"CodeMirror-linenumbers\") {\n\
        cm.display.lineGutter = gElt;\n\
        gElt.style.width = (cm.display.lineNumWidth || 1) + \"px\";\n\
      }\n\
    }\n\
    gutters.style.display = i ? \"\" : \"none\";\n\
  }\n\
\n\
  function computeMaxLength(view) {\n\
    view.maxLine = getLine(view.doc, 0); view.maxLineChanged = true;\n\
    var maxLineLength = view.maxLine.text.length;\n\
    view.doc.iter(1, view.doc.size, function(line) {\n\
      var l = line.text;\n\
      if (!line.hidden && l.length > maxLineLength) {\n\
        maxLineLength = l.length; view.maxLine = line;\n\
      }\n\
    });\n\
  }\n\
\n\
  // Make sure the gutters options contains the element\n\
  // \"CodeMirror-linenumbers\" when the lineNumbers option is true.\n\
  function setGuttersForLineNumbers(options) {\n\
    var found = false;\n\
    for (var i = 0; i < options.gutters.length; ++i) {\n\
      if (options.gutters[i] == \"CodeMirror-linenumbers\") {\n\
        if (options.lineNumbers) found = true;\n\
        else options.gutters.splice(i--, 1);\n\
      }\n\
    }\n\
    if (!found && options.lineNumbers)\n\
      options.gutters.push(\"CodeMirror-linenumbers\");\n\
  }\n\
\n\
  // SCROLLBARS\n\
\n\
  // Re-synchronize the fake scrollbars with the actual size of the\n\
  // content. Optionally force a scrollTop.\n\
  function updateScrollbars(d /* display */, docHeight, scrollTop) {\n\
    d.sizer.style.minHeight = d.heightForcer.style.top = (docHeight + 2 * paddingTop(d)) + \"px\";\n\
    var needsH = d.scroller.scrollWidth > d.scroller.clientWidth;\n\
    var needsV = d.scroller.scrollHeight > d.scroller.clientHeight;\n\
    if (needsV) {\n\
      d.scrollbarV.style.display = \"block\";\n\
      d.scrollbarV.style.bottom = needsH ? scrollbarWidth(d.measure) + \"px\" : \"0\";\n\
      d.scrollbarV.firstChild.style.height = \n\
        (d.scroller.scrollHeight - d.scroller.clientHeight + d.scrollbarV.clientHeight) + \"px\";\n\
      if (scrollTop != null) {\n\
        d.scrollbarV.scrollTop = d.scroller.scrollTop = scrollTop;\n\
        // 'Nudge' the scrollbar to work around a Webkit bug where,\n\
        // in some situations, we'd end up with a scrollbar that\n\
        // reported its scrollTop (and looked) as expected, but\n\
        // *behaved* as if it was still in a previous state (i.e.\n\
        // couldn't scroll up, even though it appeared to be at the\n\
        // bottom).\n\
        if (webkit) setTimeout(function() {\n\
          if (d.scrollbarV.scrollTop != scrollTop) return;\n\
          d.scrollbarV.scrollTop = scrollTop + (scrollTop ? -1 : 1);\n\
          d.scrollbarV.scrollTop = scrollTop;\n\
        }, 0);\n\
      }\n\
    } else d.scrollbarV.style.display = \"\";\n\
    if (needsH) {\n\
      d.scrollbarH.style.display = \"block\";\n\
      d.scrollbarH.style.right = needsV ? scrollbarWidth(d.measure) + \"px\" : \"0\";\n\
      d.scrollbarH.firstChild.style.width =\n\
        (d.scroller.scrollWidth - d.scroller.clientWidth + d.scrollbarH.clientWidth) + \"px\";\n\
    } else d.scrollbarH.style.display = \"\";\n\
    if (needsH && needsV) {\n\
      d.scrollbarFiller.style.display = \"block\";\n\
      d.scrollbarFiller.style.height = d.scrollbarFiller.style.width = scrollbarWidth(d.measure) + \"px\";\n\
    } else d.scrollbarFiller.style.display = \"\";\n\
\n\
    if (mac_geLion && scrollbarWidth(d.measure) === 0)\n\
      d.scrollbarV.style.minWidth = d.scrollbarH.style.minHeight = \"12px\";\n\
  }\n\
\n\
  function visibleLines(display, doc, scrollTop) {\n\
    var top = (scrollTop != null ? scrollTop : display.scroller.scrollTop) - paddingTop(display);\n\
    var fromHeight = Math.max(0, Math.floor(top));\n\
    var toHeight = Math.ceil(top + display.wrapper.clientHeight);\n\
    return {from: lineAtHeight(doc, fromHeight),\n\
            to: lineAtHeight(doc, toHeight)};\n\
  }\n\
\n\
  // LINE NUMBERS\n\
\n\
  function alignLineNumbers(display) {\n\
    if (display.alignWidgets) {\n\
      var margin = compensateForHScroll(display);\n\
      for (var n = display.lineDiv.firstChild; n; n = n.nextSibling)\n\
        for (var c = n.lastChild; c && c.widget; c = c.prevSibling)\n\
          if (c.widget.noHScroll)\n\
            c.style.marginLeft = (c.widget.coverGutter ? margin : margin + display.gutters.offsetWidth) + \"px\";\n\
    }\n\
    if (!display.lineGutter) return;\n\
    var l = lineNumberLeftPos(display) + \"px\";\n\
    for (var n = display.lineDiv.firstChild; n; n = n.nextSibling)\n\
      n.firstChild.style.left = l;\n\
  }\n\
\n\
  function maybeUpdateLineNumberWidth(cm) {\n\
    if (!cm.options.lineNumbers) return false;\n\
    var doc = cm.view.doc, last = lineNumberFor(cm.options, doc.size - 1), display = cm.display;\n\
    if (last.length != display.lineNumChars) {\n\
      var test = display.measure.appendChild(elt(\"div\", [elt(\"div\", last)],\n\
                                                 \"CodeMirror-linenumber CodeMirror-gutter-elt\"));\n\
      display.lineNumWidth = test.firstChild.offsetWidth;\n\
      display.lineNumChars = display.lineNumWidth ? last.length : -1;\n\
      display.lineGutter.style.width = display.lineNumWidth + \"px\";\n\
      return true;\n\
    }\n\
    return false;\n\
  }\n\
\n\
  function lineNumberFor(options, i) {\n\
    return String(options.lineNumberFormatter(i + options.firstLineNumber));\n\
  }\n\
  function compensateForHScroll(display) {\n\
    return display.scroller.getBoundingClientRect().left - display.sizer.getBoundingClientRect().left;\n\
  }\n\
  function lineNumberLeftPos(display) {\n\
    return compensateForHScroll(display) + display.lineGutter.offsetLeft;\n\
  }\n\
\n\
  // DISPLAY DRAWING\n\
\n\
  function updateDisplay(cm, changes, scrollTop) {\n\
    var oldFrom = cm.display.showingFrom, oldTo = cm.display.showingTo;\n\
    var updated = updateDisplayInner(cm, changes, scrollTop);\n\
    if (updated) {\n\
      signalLater(cm, cm, \"update\", cm);\n\
      if (cm.display.showingFrom != oldFrom || cm.display.showingTo != oldTo)\n\
        signalLater(cm, cm, \"update\", cm, cm.display.showingFrom, cm.display.showingTo);\n\
    }\n\
    updateSelection(cm);\n\
    updateScrollbars(cm.display, cm.view.doc.height, scrollTop);\n\
    return updated;\n\
  }\n\
\n\
  // Uses a set of changes plus the current scroll position to\n\
  // determine which DOM updates have to be made, and makes the\n\
  // updates.\n\
  function updateDisplayInner(cm, changes, scrollTop) {\n\
    var display = cm.display, doc = cm.view.doc;\n\
    if (!display.wrapper.clientWidth) {\n\
      display.showingFrom = display.showingTo = display.viewOffset = 0;\n\
      return;\n\
    }\n\
\n\
    // Compute the new visible window\n\
    // If scrollTop is specified, use that to determine which lines\n\
    // to render instead of the current scrollbar position.\n\
    var visible = visibleLines(display, doc, scrollTop);\n\
    // Bail out if the visible area is already rendered and nothing changed.\n\
    if (changes !== true && changes.length == 0 &&\n\
        visible.from > display.showingFrom && visible.to < display.showingTo)\n\
      return;\n\
\n\
    if (changes && changes !== true && maybeUpdateLineNumberWidth(cm))\n\
      changes = true;\n\
    display.sizer.style.marginLeft = display.scrollbarH.style.left = display.gutters.offsetWidth + \"px\";\n\
    // Used to determine which lines need their line numbers updated\n\
    var positionsChangedFrom = changes === true ? 0 : Infinity;\n\
    if (cm.options.lineNumbers && changes && changes !== true)\n\
      for (var i = 0; i < changes.length; ++i)\n\
        if (changes[i].diff) { positionsChangedFrom = changes[i].from; break; }\n\
\n\
    var from = Math.max(visible.from - 100, 0), to = Math.min(doc.size, visible.to + 100);\n\
    if (display.showingFrom < from && from - display.showingFrom < 20) from = display.showingFrom;\n\
    if (display.showingTo > to && display.showingTo - to < 20) to = Math.min(doc.size, display.showingTo);\n\
\n\
    // Create a range of theoretically intact lines, and punch holes\n\
    // in that using the change info.\n\
    var intact = changes === true ? [] :\n\
      computeIntact([{from: display.showingFrom, to: display.showingTo, domStart: 0}], changes);\n\
    // Clip off the parts that won't be visible\n\
    var intactLines = 0;\n\
    for (var i = 0; i < intact.length; ++i) {\n\
      var range = intact[i];\n\
      if (range.from < from) {range.domStart += (from - range.from); range.from = from;}\n\
      if (range.to > to) range.to = to;\n\
      if (range.from >= range.to) intact.splice(i--, 1);\n\
      else intactLines += range.to - range.from;\n\
    }\n\
    if (intactLines == to - from && from == display.showingFrom && to == display.showingTo)\n\
      return;\n\
    intact.sort(function(a, b) {return a.domStart - b.domStart;});\n\
\n\
    display.lineDiv.style.display = \"none\";\n\
    patchDisplay(cm, from, to, intact, positionsChangedFrom);\n\
    display.lineDiv.style.display = \"\";\n\
\n\
    var different = from != display.showingFrom || to != display.showingTo ||\n\
      display.lastSizeC != display.wrapper.clientHeight;\n\
    // This is just a bogus formula that detects when the editor is\n\
    // resized or the font size changes.\n\
    if (different) display.lastSizeC = display.wrapper.clientHeight;\n\
    display.showingFrom = from; display.showingTo = to;\n\
    display.viewOffset = heightAtLine(doc, from);\n\
    startWorker(cm, 100);\n\
\n\
    // Since this is all rather error prone, it is honoured with the\n\
    // only assertion in the whole file.\n\
    if (display.lineDiv.childNodes.length != display.showingTo - display.showingFrom)\n\
      throw new Error(\"BAD PATCH! \" + JSON.stringify(intact) + \" size=\" + (display.showingTo - display.showingFrom) +\n\
                      \" nodes=\" + display.lineDiv.childNodes.length);\n\
\n\
    // Update line heights for visible lines based on actual DOM\n\
    // sizes\n\
    var curNode = display.lineDiv.firstChild, heightChanged = false;\n\
    var relativeTo = curNode.offsetTop;\n\
    doc.iter(display.showingFrom, display.showingTo, function(line) {\n\
      // Work around bizarro IE7 bug where, sometimes, our curNode\n\
      // is magically replaced with a new node in the DOM, leaving\n\
      // us with a reference to an orphan (nextSibling-less) node.\n\
      if (!curNode) return;\n\
      if (!line.hidden) {\n\
        var end = curNode.offsetHeight + curNode.offsetTop;\n\
        var height = end - relativeTo, diff = line.height - height;\n\
        if (height < 2) height = textHeight(display);\n\
        relativeTo = end;\n\
        if (diff > .001 || diff < -.001) {\n\
          updateLineHeight(line, height);\n\
          heightChanged = true;\n\
        }\n\
      }\n\
      curNode = curNode.nextSibling;\n\
    });\n\
\n\
    // Position the mover div to align with the current virtual scroll position\n\
    display.mover.style.top = display.viewOffset + \"px\";\n\
    return true;\n\
  }\n\
\n\
  function computeIntact(intact, changes) {\n\
    for (var i = 0, l = changes.length || 0; i < l; ++i) {\n\
      var change = changes[i], intact2 = [], diff = change.diff || 0;\n\
      for (var j = 0, l2 = intact.length; j < l2; ++j) {\n\
        var range = intact[j];\n\
        if (change.to <= range.from && change.diff)\n\
          intact2.push({from: range.from + diff, to: range.to + diff,\n\
                        domStart: range.domStart});\n\
        else if (change.to <= range.from || change.from >= range.to)\n\
          intact2.push(range);\n\
        else {\n\
          if (change.from > range.from)\n\
            intact2.push({from: range.from, to: change.from, domStart: range.domStart});\n\
          if (change.to < range.to)\n\
            intact2.push({from: change.to + diff, to: range.to + diff,\n\
                          domStart: range.domStart + (change.to - range.from)});\n\
        }\n\
      }\n\
      intact = intact2;\n\
    }\n\
    return intact;\n\
  }\n\
\n\
  function patchDisplay(cm, from, to, intact, updateNumbersFrom) {\n\
    function killNode(node) {\n\
      var tmp = node.nextSibling;\n\
      node.parentNode.removeChild(node);\n\
      return tmp;\n\
    }\n\
    var display = cm.display, lineNumbers = cm.options.lineNumbers;\n\
    var lineNumberPos = lineNumbers && lineNumberLeftPos(display);\n\
    // The first pass removes the DOM nodes that aren't intact.\n\
    if (!intact.length) removeChildren(display.lineDiv);\n\
    else {\n\
      var domPos = 0, curNode = display.lineDiv.firstChild, n;\n\
      for (var i = 0; i < intact.length; ++i) {\n\
        var cur = intact[i];\n\
        while (cur.domStart > domPos) {curNode = killNode(curNode); domPos++;}\n\
        for (var j = cur.from, e = cur.to; j < e; ++j) {\n\
          if (lineNumbers && updateNumbersFrom <= j && curNode.firstChild)\n\
            setTextContent(curNode.firstChild, lineNumberFor(cm.options, j));\n\
          curNode = curNode.nextSibling; domPos++;\n\
        }\n\
      }\n\
      while (curNode) curNode = killNode(curNode);\n\
    }\n\
    // This pass fills in the lines that actually changed.\n\
    var nextIntact = intact.shift(), curNode = display.lineDiv.firstChild;\n\
    var j = from, gutterSpecs = cm.options.gutters;\n\
    cm.view.doc.iter(from, to, function(line) {\n\
      if (nextIntact && nextIntact.to == j) nextIntact = intact.shift();\n\
      if (!nextIntact || nextIntact.from > j) {\n\
        if (line.hidden) var lineElement = elt(\"div\");\n\
        else {\n\
          var lineElement = lineContent(cm, line), markers = line.gutterMarkers;\n\
          if (line.className) lineElement.className = line.className;\n\
          // Lines with gutter elements or a background class need\n\
          // to be wrapped again, and have the extra elements added\n\
          // to the wrapper div\n\
          if (lineNumbers || markers || line.bgClassName || (line.widgets && line.widgets.length)) {\n\
            var inside = [];\n\
            if (lineNumbers)\n\
              inside.push(elt(\"div\", lineNumberFor(cm.options, j),\n\
                              \"CodeMirror-linenumber CodeMirror-gutter-elt\",\n\
                              \"left: \" + lineNumberPos + \"px; width: \"\n\
                              + display.lineNumWidth + \"px\"));\n\
            if (markers)\n\
              for (var k = 0; k < gutterSpecs.length; ++k) {\n\
                var id = gutterSpecs[k], found = markers.hasOwnProperty(id) && markers[id];\n\
                if (found) {\n\
                  var gutterElt = display.gutters.childNodes[k];\n\
                  inside.push(elt(\"div\", [found], \"CodeMirror-gutter-elt\", \"left: \" +\n\
                                  (gutterElt.offsetLeft - display.gutters.offsetWidth) +\n\
                                  \"px; width: \" + gutterElt.clientWidth + \"px\"));\n\
                }\n\
              }\n\
            // Kludge to make sure the styled element lies behind the selection (by z-index)\n\
            if (line.bgClassName)\n\
              inside.push(elt(\"div\", \"\\u00a0\", line.bgClassName + \" CodeMirror-linebackground\"));\n\
            inside.push(lineElement);\n\
            if (line.widgets)\n\
              for (var i = 0, ws = line.widgets; i < ws.length; ++i) {\n\
                var widget = ws[i], node = elt(\"div\", [widget.node], \"CodeMirror-linewidget\");\n\
                node.widget = widget;\n\
                if (widget.noHScroll)\n\
                  node.style.width = display.wrapper.clientWidth + \"px\";\n\
                if (widget.coverGutter) {\n\
                  node.style.zIndex = 5;\n\
                  node.style.position = \"relative\";\n\
                  node.style.marginLeft =\n\
                    (widget.noHScroll ? compensateForHScroll(display) : -display.gutters.offsetWidth) + \"px\";\n\
                }\n\
                inside.push(node);\n\
              }\n\
            lineElement = elt(\"div\", inside, null, \"position: relative\");\n\
            if (ie_lt8) lineElement.style.zIndex = 2;\n\
          }\n\
        }\n\
        display.lineDiv.insertBefore(lineElement, curNode);\n\
      } else {\n\
        curNode = curNode.nextSibling;\n\
      }\n\
      ++j;\n\
    });\n\
  }\n\
\n\
  // SELECTION / CURSOR\n\
\n\
  function selHead(view) {\n\
    return view.sel.inverted ? view.sel.from : view.sel.to;\n\
  }\n\
\n\
  function updateSelection(cm) {\n\
    var headPos, display = cm.display, doc = cm.view.doc, sel = cm.view.sel;\n\
    if (posEq(sel.from, sel.to)) { // No selection, single cursor\n\
      var pos = headPos = cursorCoords(cm, sel.from, \"div\");\n\
      display.cursor.style.left = pos.left + \"px\";\n\
      display.cursor.style.top = pos.top + \"px\";\n\
      display.cursor.style.height = (pos.bottom - pos.top) * .85 + \"px\";\n\
      display.cursor.style.display = \"\";\n\
      display.selectionDiv.style.display = \"none\";\n\
\n\
      if (pos.other) {\n\
        display.otherCursor.style.display = \"\";\n\
        display.otherCursor.style.left = pos.other.left + \"px\";\n\
        display.otherCursor.style.top = pos.other.top + \"px\";\n\
        display.otherCursor.style.height = (pos.other.bottom - pos.other.top) * .85 + \"px\";\n\
      } else { display.otherCursor.style.display = \"none\"; }\n\
    } else {\n\
      headPos = cursorCoords(cm, selHead(cm.view), \"div\");\n\
      var fragment = document.createDocumentFragment();\n\
      var clientWidth = display.lineSpace.clientWidth;\n\
      var add = function(left, top, width, bottom) {\n\
        if (top < 0) top = 0;\n\
        fragment.appendChild(elt(\"div\", null, \"CodeMirror-selected\", \"position: absolute; left: \" + left +\n\
                                 \"px; top: \" + top + \"px; width: \" + (width == null ? clientWidth : width) +\n\
                                 \"px; height: \" + (bottom - top) + \"px\"));\n\
      };\n\
\n\
      var middleFrom = sel.from.line + 1, middleTo = sel.to.line - 1, sameLine = sel.from.line == sel.to.line;\n\
      var drawForLine = function(line, from, toArg, retTop) {\n\
        var lineObj = getLine(doc, line), lineLen = lineObj.text.length;\n\
        var coords = function(ch) { return charCoords(cm, {line: line, ch: ch}, \"div\", lineObj); };\n\
        var rVal = retTop ? Infinity : -Infinity;\n\
        iterateBidiSections(getOrder(lineObj), from, toArg == null ? lineLen : toArg, function(from, to, dir) {\n\
          var leftPos = coords(dir == \"rtl\" ? to - 1 : from);\n\
          var rightPos = coords(dir == \"rtl\" ? from : to - 1);\n\
          var left = leftPos.left, right = rightPos.right;\n\
          if (rightPos.top - leftPos.top > 3) { // Different lines, draw top part\n\
            add(left, leftPos.top, null, leftPos.bottom);\n\
            left = paddingLeft(display);\n\
            if (leftPos.bottom < rightPos.top) add(left, leftPos.bottom, null, rightPos.top);\n\
          }\n\
          if (toArg == null && to == lineLen) right = clientWidth;\n\
          rVal = retTop ? Math.min(rightPos.top, rVal) : Math.max(rightPos.bottom, rVal);\n\
          add(left, rightPos.top, right - left, rightPos.bottom);\n\
        });\n\
        return rVal;\n\
      };\n\
\n\
      var middleTop = Infinity, middleBot = -Infinity;\n\
      if (sel.from.ch || sameLine)\n\
        // Draw the first line of selection.\n\
        middleTop = drawForLine(sel.from.line, sel.from.ch, sameLine ? sel.to.ch : null);\n\
      else\n\
        // Simply include it in the middle block.\n\
        middleFrom = sel.from.line;\n\
\n\
      if (!sameLine && sel.to.ch)\n\
        middleBot = drawForLine(sel.to.line, 0, sel.to.ch, true);\n\
\n\
      if (middleFrom <= middleTo) {\n\
        // Draw the middle\n\
        var botLine = getLine(doc, middleTo),\n\
        bottom = charCoords(cm, {line: middleTo, ch: botLine.text.length}, \"div\", botLine);\n\
        // Kludge to try and prevent fetching coordinates twice if\n\
        // start end end are on same line.\n\
        var top = (middleFrom != middleTo || botLine.height > bottom.bottom - bottom.top) ?\n\
          charCoords(cm, {line: middleFrom, ch: 0}, \"div\") : bottom;\n\
        middleTop = Math.min(middleTop, top.top);\n\
        middleBot = Math.max(middleBot, bottom.bottom);\n\
      }\n\
      if (middleTop < middleBot) add(paddingLeft(display), middleTop, null, middleBot);\n\
\n\
      removeChildrenAndAdd(display.selectionDiv, fragment);\n\
      display.cursor.style.display = display.otherCursor.style.display = \"none\";\n\
      display.selectionDiv.style.display = \"\";\n\
    }\n\
\n\
    // Move the hidden textarea near the cursor to prevent scrolling artifacts\n\
    var wrapOff = display.wrapper.getBoundingClientRect(), lineOff = display.lineDiv.getBoundingClientRect();\n\
    display.inputDiv.style.top = Math.max(0, Math.min(display.wrapper.clientHeight - 10,\n\
                                                      headPos.top + lineOff.top - wrapOff.top)) + \"px\";\n\
    display.inputDiv.style.left = Math.max(0, Math.min(display.wrapper.clientWidth - 10,\n\
                                                       headPos.left + lineOff.left - wrapOff.left)) + \"px\";\n\
  }\n\
\n\
  // Cursor-blinking\n\
  function restartBlink(cm) {\n\
    var display = cm.display;\n\
    clearInterval(display.blinker);\n\
    var on = true;\n\
    display.cursor.style.visibility = display.otherCursor.style.visibility = \"\";\n\
    display.blinker = setInterval(function() {\n\
      display.cursor.style.visibility = display.otherCursor.style.visibility = (on = !on) ? \"\" : \"hidden\";\n\
    }, cm.options.cursorBlinkRate);\n\
  }\n\
\n\
  // HIGHLIGHT WORKER\n\
\n\
  function startWorker(cm, time) {\n\
    if (cm.view.doc.frontier < cm.display.showingTo)\n\
      cm.view.doc.highlight.set(time, bind(highlightWorker, cm));\n\
  }\n\
\n\
  function highlightWorker(cm) {\n\
    var doc = cm.view.doc;\n\
    if (doc.frontier >= cm.display.showingTo) return;\n\
    var end = +new Date + cm.options.workTime;\n\
    var state = copyState(doc.mode, getStateBefore(doc, doc.frontier));\n\
    var startFrontier = doc.frontier;\n\
    doc.iter(doc.frontier, cm.display.showingTo, function(line) {\n\
      if (doc.frontier >= cm.display.showingFrom) { // Visible\n\
        line.highlight(doc.mode, state, cm.options.tabSize);\n\
        line.stateAfter = copyState(doc.mode, state);\n\
      } else {\n\
        line.process(doc.mode, state, cm.options.tabSize);\n\
        line.stateAfter = doc.frontier % 5 == 0 ? copyState(doc.mode, state) : null;\n\
      }\n\
      ++doc.frontier;\n\
      if (+new Date > end) {\n\
        startWorker(cm, cm.options.workDelay);\n\
        return true;\n\
      }\n\
    });\n\
    if (cm.display.showingTo > startFrontier && doc.frontier >= cm.display.showingFrom)\n\
      operation(cm, function() {regChange(this, startFrontier, doc.frontier);})();\n\
  }\n\
\n\
  // Finds the line to start with when starting a parse. Tries to\n\
  // find a line with a stateAfter, so that it can start with a\n\
  // valid state. If that fails, it returns the line with the\n\
  // smallest indentation, which tends to need the least context to\n\
  // parse correctly.\n\
  function findStartLine(doc, n) {\n\
    var minindent, minline;\n\
    for (var search = n, lim = n - 100; search > lim; --search) {\n\
      if (search == 0) return 0;\n\
      var line = getLine(doc, search-1);\n\
      if (line.stateAfter) return search;\n\
      var indented = line.indentation(doc.tabSize);\n\
      if (minline == null || minindent > indented) {\n\
        minline = search - 1;\n\
        minindent = indented;\n\
      }\n\
    }\n\
    return minline;\n\
  }\n\
\n\
  function getStateBefore(doc, n) {\n\
    var pos = findStartLine(doc, n), state = pos && getLine(doc, pos-1).stateAfter;\n\
    if (!state) state = startState(doc.mode);\n\
    else state = copyState(doc.mode, state);\n\
    doc.iter(pos, n, function(line) {\n\
      line.process(doc.mode, state, doc.tabSize);\n\
      line.stateAfter = (pos == n - 1 || pos % 5 == 0) ? copyState(doc.mode, state) : null;\n\
    });\n\
    return state;\n\
  }\n\
\n\
  // POSITION MEASUREMENT\n\
  \n\
  function paddingTop(display) {return display.lineSpace.offsetTop;}\n\
  function paddingLeft(display) {\n\
    var e = removeChildrenAndAdd(display.measure, elt(\"pre\")).appendChild(elt(\"span\", \"x\"));\n\
    return e.offsetLeft;\n\
  }\n\
\n\
  function measureLine(cm, line, ch) {\n\
    var wrapping = cm.options.lineWrapping, display = cm.display;\n\
    // First look in the cache\n\
    var cache = cm.display.measureLineCache;\n\
    for (var i = 0; i < cache.length; ++i) {\n\
      var memo = cache[i];\n\
      if (memo.ch == ch && memo.text == line.text &&\n\
          (!wrapping || display.scroller.clientWidth == memo.width))\n\
        return {top: memo.top, bottom: memo.bottom, left: memo.left, right: memo.right};\n\
    }\n\
\n\
    var atEnd = ch && ch == line.text.length;\n\
    var pre = lineContent(cm, line, atEnd ? ch - 1 : ch);\n\
    removeChildrenAndAdd(display.measure, pre);\n\
    var anchor = pre.anchor, outer = display.lineDiv.getBoundingClientRect();\n\
    // We'll sample once at the top, once at the bottom of the line,\n\
    // to get the real line height (in case there tokens on the line\n\
    // with bigger fonts)\n\
    anchor.style.verticalAlign = \"top\";\n\
    var box1 = anchor.getBoundingClientRect(), left = box1.left - outer.left, right = box1.right - outer.left;\n\
    if (ie) {\n\
      var left1 = anchor.offsetLeft;\n\
      // In IE, verticalAlign does not influence offsetTop, unless\n\
      // the element is an inline-block. Unfortunately, inline\n\
      // blocks have different wrapping behaviour, so we have to do\n\
      // some icky thing with inserting \"Zero-Width No-Break Spaces\"\n\
      // to compensate for wrapping artifacts.\n\
      anchor.style.display = \"inline-block\";\n\
      if (wrapping && anchor.offsetLeft != left1) {\n\
        anchor.parentNode.insertBefore(document.createTextNode(\"\\ufeff\"), anchor);\n\
        if (anchor.offsetLeft != left1)\n\
          anchor.parentNode.insertBefore(document.createTextNode(\"\\ufeff\"), anchor.nextSibling);\n\
        if (anchor.offsetLeft != left1)\n\
          anchor.parentNode.removeChild(anchor.previousSibling);\n\
      }\n\
    }\n\
    var top = Math.max(0, box1.top - outer.top);\n\
    anchor.style.verticalAlign = \"bottom\";\n\
    var bottom = Math.min(anchor.getBoundingClientRect().bottom - outer.top, pre.offsetHeight);\n\
    if (atEnd) left = right;\n\
\n\
    // Store result in the cache\n\
    var memo = {ch: ch, text: line.text, width: display.wrapper.clientWidth,\n\
                top: top, bottom: bottom, left: left, right: right};\n\
    if (cache.length == 8) cache[++cache.pos % 8] = memo;\n\
    else cache.push(memo);\n\
\n\
    return {top: top, bottom: bottom, left: left, right: right};\n\
  }\n\
\n\
  // Context is one of \"line\", \"div\" (display.lineDiv), \"local\"/null (editor), or \"page\"\n\
  function intoCoordSystem(cm, pos, rect, context) {\n\
    if (context == \"line\") return rect;\n\
    if (!context) context = \"local\";\n\
    var yOff = heightAtLine(cm.view.doc, pos.line);\n\
    if (context != \"local\") yOff -= cm.display.viewOffset;\n\
    if (context == \"page\") {\n\
      var lOff = cm.display.lineSpace.getBoundingClientRect();\n\
      yOff += lOff.top; rect.left += lOff.left; rect.right += lOff.right;\n\
    }\n\
    rect.top += yOff; rect.bottom += yOff;\n\
    return rect;\n\
  }\n\
\n\
  function charCoords(cm, pos, context, lineObj) {\n\
    if (!lineObj) lineObj = getLine(cm.view.doc, pos.line);\n\
    return intoCoordSystem(cm, pos, measureLine(cm, lineObj, pos.ch), context);\n\
  }\n\
\n\
  function cursorCoords(cm, pos, context, lineObj) {\n\
    lineObj = lineObj || getLine(cm.view.doc, pos.line);\n\
    function get(ch, right) {\n\
      var m = measureLine(cm, lineObj, ch);\n\
      if (right) m.left = m.right; else m.right = m.left;\n\
      return intoCoordSystem(cm, pos, m, context);\n\
    }\n\
    var order = getOrder(lineObj), ch = pos.ch;\n\
    if (!order) return get(ch);\n\
    var main, other, linedir = order[0].level;\n\
    for (var i = 0; i < order.length; ++i) {\n\
      var part = order[i], rtl = part.level % 2, nb, here;\n\
      if (part.from < ch && part.to > ch) return get(ch, rtl);\n\
      var left = rtl ? part.to : part.from, right = rtl ? part.from : part.to;\n\
      if (left == ch) {\n\
        // Opera and IE return bogus offsets and widths for edges\n\
        // where the direction flips, but only for the side with the\n\
        // lower level. So we try to use the side with the higher\n\
        // level.\n\
        if (i && part.level < (nb = order[i-1]).level) here = get(nb.level % 2 ? nb.from : nb.to - 1, true);\n\
        else here = get(rtl && part.from != part.to ? ch - 1 : ch);\n\
        if (rtl == linedir) main = here; else other = here;\n\
      } else if (right == ch) {\n\
        var nb = i < order.length - 1 && order[i+1];\n\
        if (!rtl && nb && nb.from == nb.to) continue;\n\
        if (nb && part.level < nb.level) here = get(nb.level % 2 ? nb.to - 1 : nb.from);\n\
        else here = get(rtl ? ch : ch - 1, true);\n\
        if (rtl == linedir) main = here; else other = here;\n\
      }\n\
    }\n\
    if (linedir && !ch) other = get(order[0].to - 1);\n\
    if (!main) return other;\n\
    if (other) main.other = other;\n\
    return main;\n\
  }\n\
\n\
  // Coords must be lineSpace-local\n\
  function coordsChar(cm, x, y) {\n\
    var display = cm.display, doc = cm.view.doc;\n\
    var cw = charWidth(display), heightPos = display.viewOffset + y;\n\
    if (heightPos < 0) return {line: 0, ch: 0};\n\
    var lineNo = lineAtHeight(doc, heightPos);\n\
    if (lineNo >= doc.size) return {line: doc.size - 1, ch: getLine(doc, doc.size - 1).text.length};\n\
    var lineObj = getLine(doc, lineNo);\n\
    if (!lineObj.text.length) return {line: lineNo, ch: 0};\n\
    var tw = cm.options.lineWrapping, innerOff = tw ? heightPos - heightAtLine(doc, lineNo) : 0;\n\
    if (x < 0) x = 0;\n\
    var wrongLine = false;\n\
    function getX(ch) {\n\
      var sp = cursorCoords(cm, {line: lineNo, ch: ch}, \"line\", lineObj);\n\
      if (tw) {\n\
        wrongLine = true;\n\
        if (innerOff > sp.bottom) return Math.max(0, sp.left - display.wrapper.clientWidth);\n\
        else if (innerOff < sp.top) return sp.left + display.wrapper.clientWidth;\n\
        else wrongLine = false;\n\
      }\n\
      return sp.left;\n\
    }\n\
    var bidi = getOrder(lineObj), dist = lineObj.text.length;\n\
    var from = lineLeft(lineObj), fromX = 0, to = lineRight(lineObj), toX;\n\
    if (!bidi) {\n\
      // Guess a suitable upper bound for our search.\n\
      var estimated = Math.min(to, Math.ceil((x + Math.floor(innerOff / textHeight(display)) *\n\
                                              display.wrapper.clientWidth * .9) / cw));\n\
      for (;;) {\n\
        var estX = getX(estimated);\n\
        if (estX <= x && estimated < to) estimated = Math.min(to, Math.ceil(estimated * 1.2));\n\
        else {toX = estX; to = estimated; break;}\n\
      }\n\
      // Try to guess a suitable lower bound as well.\n\
      estimated = Math.floor(to * 0.8); estX = getX(estimated);\n\
      if (estX < x) {from = estimated; fromX = estX;}\n\
      dist = to - from;\n\
    } else toX = getX(to);\n\
    if (x > toX) return {line: lineNo, ch: to};\n\
    // Do a binary search between these bounds.\n\
    for (;;) {\n\
      if (bidi ? to == from || to == moveVisually(lineObj, from, 1) : to - from <= 1) {\n\
        var after = x - fromX < toX - x, ch = after ? from : to;\n\
        while (isExtendingChar.test(lineObj.text.charAt(ch))) ++ch;\n\
        return {line: lineNo, ch: ch, after: after};\n\
      }\n\
      var step = Math.ceil(dist / 2), middle = from + step;\n\
      if (bidi) {\n\
        middle = from;\n\
        for (var i = 0; i < step; ++i) middle = moveVisually(lineObj, middle, 1);\n\
      }\n\
      var middleX = getX(middle);\n\
      if (middleX > x) {to = middle; toX = middleX; if (wrongLine) toX += 1000; dist -= step;}\n\
      else {from = middle; fromX = middleX; dist = step;}\n\
    }\n\
  }\n\
\n\
  var measureText;\n\
  function textHeight(display) {\n\
    if (display.cachedTextHeight != null) return display.cachedTextHeight;\n\
    if (measureText == null) {\n\
      measureText = elt(\"pre\");\n\
      // Measure a bunch of lines, for browsers that compute\n\
      // fractional heights.\n\
      for (var i = 0; i < 49; ++i) {\n\
        measureText.appendChild(document.createTextNode(\"x\"));\n\
        measureText.appendChild(elt(\"br\"));\n\
      }\n\
      measureText.appendChild(document.createTextNode(\"x\"));\n\
    }\n\
    removeChildrenAndAdd(display.measure, measureText);\n\
    var height = measureText.offsetHeight / 50;\n\
    if (height > 3) display.cachedTextHeight = height;\n\
    removeChildren(display.measure);\n\
    return height || 1;\n\
  }\n\
\n\
  function charWidth(display) {\n\
    if (display.cachedCharWidth != null) return display.cachedCharWidth;\n\
    var anchor = elt(\"span\", \"x\");\n\
    var pre = elt(\"pre\", [anchor]);\n\
    removeChildrenAndAdd(display.measure, pre);\n\
    var width = anchor.offsetWidth;\n\
    if (width > 2) display.cachedCharWidth = width;\n\
    return width || 10;\n\
  }\n\
\n\
  // OPERATIONS\n\
\n\
  // Operations are used to wrap changes in such a way that each\n\
  // change won't have to update the cursor and display (which would\n\
  // be awkward, slow, and error-prone), but instead updates are\n\
  // batched and then all combined and executed at once.\n\
\n\
  function startOperation(cm) {\n\
    if (cm.curOp) ++cm.curOp.depth;\n\
    else cm.curOp = {\n\
      // Nested operations delay update until the outermost one\n\
      // finishes.\n\
      depth: 1,\n\
      // An array of ranges of lines that have to be updated. See\n\
      // updateDisplay.\n\
      changes: [],\n\
      delayedCallbacks: [],\n\
      updateInput: null,\n\
      userSelChange: null,\n\
      textChanged: null,\n\
      selectionChanged: false,\n\
      updateMaxLine: false\n\
    };\n\
  }\n\
\n\
  function endOperation(cm) {\n\
    var op = cm.curOp;\n\
    if (--op.depth) return;\n\
    cm.curOp = null;\n\
    var view = cm.view, display = cm.display;\n\
    if (op.updateMaxLine) computeMaxLength(view);\n\
    if (view.maxLineChanged && !cm.options.lineWrapping) {\n\
      var width = measureLine(cm, view.maxLine, view.maxLine.text.length).left;\n\
      display.sizer.style.minWidth = (width + 3 + scrollerCutOff) + \"px\";\n\
      view.maxLineChanged = false;\n\
    }\n\
    var newScrollPos, updated;\n\
    if (op.selectionChanged) {\n\
      var coords = cursorCoords(cm, selHead(view));\n\
      newScrollPos = calculateScrollPos(display, coords.left, coords.top, coords.left, coords.bottom);\n\
    }\n\
    if (op.changes.length || newScrollPos && newScrollPos.scrollTop != null)\n\
      updated = updateDisplay(cm, op.changes, newScrollPos && newScrollPos.scrollTop);\n\
    if (!updated && op.selectionChanged) updateSelection(cm);\n\
    if (newScrollPos) scrollCursorIntoView(cm);\n\
    if (op.selectionChanged) restartBlink(cm);\n\
\n\
    if (view.focused && op.updateInput)\n\
      resetInput(cm, op.userSelChange);\n\
\n\
    if (op.textChanged)\n\
      signal(cm, \"change\", cm, op.textChanged);\n\
    if (op.selectionChanged) signal(cm, \"cursorActivity\", cm);\n\
    for (var i = 0; i < op.delayedCallbacks.length; ++i) op.delayedCallbacks[i](cm);\n\
  }\n\
\n\
  // Wraps a function in an operation. Returns the wrapped function.\n\
  function operation(cm1, f) {\n\
    return function() {\n\
      var cm = cm1 || this;\n\
      startOperation(cm);\n\
      try {var result = f.apply(cm, arguments);}\n\
      finally {endOperation(cm);}\n\
      return result;\n\
    };\n\
  }\n\
\n\
  function regChange(cm, from, to, lendiff) {\n\
    cm.curOp.changes.push({from: from, to: to, diff: lendiff});\n\
  }\n\
\n\
  function compoundChange(cm, f) {\n\
    var hist = cm.view.doc.history;\n\
    hist.startCompound();\n\
    try { return f(); } finally { hist.endCompound(); }\n\
  }\n\
\n\
  // INPUT HANDLING\n\
\n\
  function slowPoll(cm) {\n\
    if (cm.view.pollingFast) return;\n\
    cm.display.poll.set(cm.options.pollInterval, function() {\n\
      readInput(cm);\n\
      if (cm.view.focused) slowPoll(cm);\n\
    });\n\
  }\n\
\n\
  function fastPoll(cm) {\n\
    var missed = false;\n\
    cm.display.pollingFast = true;\n\
    function p() {\n\
      var changed = readInput(cm);\n\
      if (!changed && !missed) {missed = true; cm.display.poll.set(60, p);}\n\
      else {cm.display.pollingFast = false; slowPoll(cm);}\n\
    }\n\
    cm.display.poll.set(20, p);\n\
  }\n\
\n\
  // prevInput is a hack to work with IME. If we reset the textarea\n\
  // on every change, that breaks IME. So we look for changes\n\
  // compared to the previous content instead. (Modern browsers have\n\
  // events that indicate IME taking place, but these are not widely\n\
  // supported or compatible enough yet to rely on.)\n\
  function readInput(cm) {\n\
    var input = cm.display.input, prevInput = cm.display.prevInput, view = cm.view, sel = view.sel;\n\
    if (!view.focused || hasSelection(input) || cm.options.readOnly) return false;\n\
    var text = input.value;\n\
    if (text == prevInput && posEq(sel.from, sel.to)) return false;\n\
    startOperation(cm);\n\
    view.sel.shift = null;\n\
    var same = 0, l = Math.min(prevInput.length, text.length);\n\
    while (same < l && prevInput[same] == text[same]) ++same;\n\
    if (same < prevInput.length)\n\
      sel.from = {line: sel.from.line, ch: sel.from.ch - (prevInput.length - same)};\n\
    else if (view.overwrite && posEq(sel.from, sel.to))\n\
      sel.to = {line: sel.to.line, ch: Math.min(getLine(cm.view.doc, sel.to.line).text.length, sel.to.ch + (text.length - same))};\n\
    var updateInput = cm.curOp.updateInput;\n\
    cm.replaceSelection(text.slice(same), \"end\");\n\
    cm.curOp.updateInput = updateInput;\n\
    if (text.length > 1000) { input.value = cm.display.prevInput = \"\"; }\n\
    else cm.display.prevInput = text;\n\
    endOperation(cm);\n\
    return true;\n\
  }\n\
\n\
  function resetInput(cm, user) {\n\
    var view = cm.view, minimal, selected;\n\
    if (!posEq(view.sel.from, view.sel.to)) {\n\
      cm.display.prevInput = \"\";\n\
      minimal = hasCopyEvent &&\n\
        (view.sel.to.line - view.sel.from.line > 100 || (selected = cm.getSelection()).length > 1000);\n\
      if (minimal) cm.display.input.value = \"-\";\n\
      else cm.display.input.value = selected || cm.getSelection();\n\
      if (view.focused) selectInput(cm.display.input);\n\
    } else if (user) cm.display.prevInput = cm.display.input.value = \"\";\n\
    cm.display.inaccurateSelection = minimal;\n\
  }\n\
\n\
  function focusInput(cm) {\n\
    if (cm.options.readOnly != \"nocursor\") cm.display.input.focus();\n\
  }\n\
\n\
  // EVENT HANDLERS\n\
\n\
  function registerEventHandlers(cm) {\n\
    var d = cm.display;\n\
    on(d.scroller, \"mousedown\", operation(cm, onMouseDown));\n\
    on(d.gutters, \"mousedown\", operation(cm, clickInGutter));\n\
    on(d.scroller, \"dblclick\", operation(cm, e_preventDefault));\n\
    on(d.lineSpace, \"selectstart\", function(e) {\n\
      if (!mouseEventInWidget(d, e)) e_preventDefault(e);\n\
    });\n\
    // Gecko browsers fire contextmenu *after* opening the menu, at\n\
    // which point we can't mess with it anymore. Context menu is\n\
    // handled in onMouseDown for Gecko.\n\
    if (!gecko) on(d.scroller, \"contextmenu\", function(e) {onContextMenu(cm, e);});\n\
\n\
    on(d.scroller, \"scroll\", function() {\n\
      if (d.scroller.scrollTop != cm.view.scrollTop) {\n\
        d.scrollbarV.scrollTop = cm.view.scrollTop = d.scroller.scrollTop;\n\
        updateDisplay(cm, []);\n\
      }\n\
      if (d.scroller.scrollLeft != cm.view.scrollLeft) {\n\
        d.scrollbarH.scrollLeft = cm.view.scrollLeft = d.scroller.scrollLeft;\n\
        alignLineNumbers(cm.display);\n\
      }\n\
      signal(cm, \"scroll\", cm);\n\
    });\n\
    on(d.scrollbarV, \"scroll\", function() {\n\
      if (d.scrollbarV.scrollTop != cm.view.scrollTop) {\n\
        d.scroller.scrollTop = cm.view.scrollTop = d.scrollbarV.scrollTop;\n\
        updateDisplay(cm, []);\n\
      }\n\
    });\n\
    on(d.scrollbarH, \"scroll\", function() {\n\
      if (d.scrollbarH.scrollLeft != cm.view.scrollLeft) {\n\
        d.scroller.scrollLeft = cm.view.scrollLeft = d.scrollbarH.scrollLeft;\n\
        alignLineNumbers(cm.display);\n\
      }\n\
    });\n\
\n\
    function reFocus() { if (cm.view.focused) setTimeout(bind(focusInput, cm), 0); }\n\
    on(d.scrollbarH, \"mousedown\", reFocus);\n\
    on(d.scrollbarV, \"mousedown\", reFocus);\n\
    // Prevent wrapper from ever scrolling\n\
    on(d.wrapper, \"scroll\", function() { d.wrapper.scrollTop = d.wrapper.scrollLeft = 0; });\n\
    on(window, \"resize\", function resizeHandler() {\n\
      // Might be a text scaling operation, clear size caches.\n\
      d.cachedCharWidth = d.cachedTextHeight = null;\n\
      d.measureLineCache.length = d.measureLineCache.pos = 0;\n\
      if (d.wrapper.parentNode) updateDisplay(cm, true);\n\
      else off(window, \"resize\", resizeHandler);\n\
    });\n\
\n\
    on(d.input, \"keyup\", operation(cm, function(e) {\n\
      if (cm.options.onKeyEvent && cm.options.onKeyEvent(cm, addStop(e))) return;\n\
      if (e_prop(e, \"keyCode\") == 16) cm.view.sel.shift = null;\n\
    }));\n\
    on(d.input, \"input\", bind(fastPoll, cm));\n\
    on(d.input, \"keydown\", operation(cm, onKeyDown));\n\
    on(d.input, \"keypress\", operation(cm, onKeyPress));\n\
    on(d.input, \"focus\", bind(onFocus, cm));\n\
    on(d.input, \"blur\", bind(onBlur, cm));\n\
\n\
    function drag_(e) {\n\
      if (cm.options.onDragEvent && cm.options.onDragEvent(cm, addStop(e))) return;\n\
      e_stop(e);\n\
    }\n\
    if (cm.options.dragDrop) {\n\
      on(d.scroller, \"dragstart\", function(e){onDragStart(cm, e);});\n\
      on(d.scroller, \"dragenter\", drag_);\n\
      on(d.scroller, \"dragover\", drag_);\n\
      on(d.scroller, \"drop\", operation(cm, onDrop));\n\
    }\n\
    on(d.scroller, \"paste\", function(){focusInput(cm); fastPoll(cm);});\n\
    on(d.input, \"paste\", bind(fastPoll, cm));\n\
    function prepareCopy() {\n\
      if (d.inaccurateSelection) {\n\
        d.prevInput = \"\";\n\
        d.inaccurateSelection = false;\n\
        d.input.value = cm.getSelection();\n\
        selectInput(d.input);\n\
      }\n\
    }\n\
    on(d.input, \"cut\", prepareCopy);\n\
    on(d.input, \"copy\", prepareCopy);\n\
\n\
    // Needed to handle Tab key in KHTML\n\
    if (khtml) on(d.sizer, \"mouseup\", function() {\n\
        if (document.activeElement == d.input) d.input.blur();\n\
        focusInput(cm);\n\
    });\n\
  }\n\
\n\
  function mouseEventInWidget(display, e) {\n\
    for (var n = e_target(e); n != display.wrapper; n = n.parentNode)\n\
      if (/\\bCodeMirror-linewidget\\b/.test(n.className) ||\n\
          n.parentNode == display.sizer && n != display.mover) return true;\n\
  }\n\
\n\
  function posFromMouse(cm, e, liberal) {\n\
    var display = cm.display;\n\
    if (!liberal) {\n\
      var target = e_target(e);\n\
      if (target == display.scrollbarH || target == display.scrollbarH.firstChild ||\n\
          target == display.scrollbarV || target == display.scrollbarV.firstChild ||\n\
          target == display.scrollbarFiller) return null;\n\
    }\n\
    var x, y, space = display.lineSpace.getBoundingClientRect();\n\
    // Fails unpredictably on IE[67] when mouse is dragged around quickly.\n\
    try { x = e.clientX; y = e.clientY; } catch (e) { return null; }\n\
    return coordsChar(cm, x - space.left, y - space.top);\n\
  }\n\
\n\
  var lastClick, lastDoubleClick;\n\
  function onMouseDown(e) {\n\
    var cm = this, display = cm.display, view = cm.view, sel = view.sel, doc = view.doc;\n\
    setShift(cm.view, e_prop(e, \"shiftKey\"));\n\
\n\
    if (mouseEventInWidget(display, e)) {\n\
      if (!webkit) {\n\
        display.scroller.draggable = false;\n\
        setTimeout(function(){display.scroller.draggable = true;}, 100);\n\
      }\n\
      return;\n\
    }\n\
    if (clickInGutter.call(cm, e)) return;\n\
    var start = posFromMouse(cm, e);\n\
\n\
    switch (e_button(e)) {\n\
    case 3:\n\
      if (gecko) onContextMenu.call(cm, cm, e);\n\
      return;\n\
    case 2:\n\
      if (start) setSelectionUser(cm, start, start);\n\
      setTimeout(bind(focusInput, cm), 20);\n\
      e_preventDefault(e);\n\
      return;\n\
    }\n\
    // For button 1, if it was clicked inside the editor\n\
    // (posFromMouse returning non-null), we have to adjust the\n\
    // selection.\n\
    if (!start) {if (e_target(e) == display.scroller) e_preventDefault(e); return;}\n\
\n\
    if (!view.focused) onFocus(cm);\n\
\n\
    var now = +new Date, type = \"single\";\n\
    if (lastDoubleClick && lastDoubleClick.time > now - 400 && posEq(lastDoubleClick.pos, start)) {\n\
      type = \"triple\";\n\
      e_preventDefault(e);\n\
      setTimeout(bind(focusInput, cm), 20);\n\
      selectLine(cm, start.line);\n\
    } else if (lastClick && lastClick.time > now - 400 && posEq(lastClick.pos, start)) {\n\
      type = \"double\";\n\
      lastDoubleClick = {time: now, pos: start};\n\
      e_preventDefault(e);\n\
      var word = findWordAt(getLine(doc, start.line).text, start);\n\
      setSelectionUser(cm, word.from, word.to);\n\
    } else { lastClick = {time: now, pos: start}; }\n\
\n\
    var last = start;\n\
    if (cm.options.dragDrop && dragAndDrop && !cm.options.readOnly && !posEq(sel.from, sel.to) &&\n\
        !posLess(start, sel.from) && !posLess(sel.to, start) && type == \"single\") {\n\
      var dragEnd = operation(cm, function(e2) {\n\
        if (webkit) display.scroller.draggable = false;\n\
        view.draggingText = false;\n\
        off(document, \"mouseup\", dragEnd);\n\
        off(display.scroller, \"drop\", dragEnd);\n\
        if (Math.abs(e.clientX - e2.clientX) + Math.abs(e.clientY - e2.clientY) < 10) {\n\
          e_preventDefault(e2);\n\
          setSelectionUser(cm, start, start);\n\
          focusInput(cm);\n\
        }\n\
      });\n\
      // Let the drag handler handle this.\n\
      if (webkit) display.scroller.draggable = true;\n\
      view.draggingText = true;\n\
      // IE's approach to draggable\n\
      if (display.scroller.dragDrop) display.scroller.dragDrop();\n\
      on(document, \"mouseup\", dragEnd);\n\
      on(display.scroller, \"drop\", dragEnd);\n\
      return;\n\
    }\n\
    e_preventDefault(e);\n\
    if (type == \"single\") setSelectionUser(cm, start, start);\n\
\n\
    var startstart = sel.from, startend = sel.to;\n\
\n\
    function doSelect(cur) {\n\
      if (type == \"single\") {\n\
        setSelectionUser(cm, start, cur);\n\
      } else if (type == \"double\") {\n\
        var word = findWordAt(getLine(doc, cur.line).text, cur);\n\
        if (posLess(cur, startstart)) setSelectionUser(cm, word.from, startend);\n\
        else setSelectionUser(cm, startstart, word.to);\n\
      } else if (type == \"triple\") {\n\
        if (posLess(cur, startstart)) setSelectionUser(cm, startend, clipPos(doc, {line: cur.line, ch: 0}));\n\
        else setSelectionUser(cm, startstart, clipPos(doc, {line: cur.line + 1, ch: 0}));\n\
      }\n\
    }\n\
\n\
    var editorSize = display.wrapper.getBoundingClientRect();\n\
    // Used to ensure timeout re-tries don't fire when another extend\n\
    // happened in the meantime (clearTimeout isn't reliable -- at\n\
    // least on Chrome, the timeouts still happen even when cleared,\n\
    // if the clear happens after their scheduled firing time).\n\
    var counter = 0;\n\
\n\
    function extend(e) {\n\
      var curCount = ++counter;\n\
      var cur = posFromMouse(cm, e, true);\n\
      if (!cur) return;\n\
      if (!posEq(cur, last)) {\n\
        if (!view.focused) onFocus(cm);\n\
        last = cur;\n\
        doSelect(cur);\n\
        var visible = visibleLines(display, doc);\n\
        if (cur.line >= visible.to || cur.line < visible.from)\n\
          setTimeout(operation(cm, function(){if (counter == curCount) extend(e);}), 150);\n\
      } else {\n\
        var outside = e.clientY < editorSize.top ? -20 : e.clientY > editorSize.bottom ? 20 : 0;\n\
        if (outside) setTimeout(operation(cm, function() {\n\
          if (counter != curCount) return;\n\
          display.scroller.scrollTop += outside;\n\
          extend(e);\n\
        }), 50);\n\
      }\n\
    }\n\
\n\
    function done(e) {\n\
      counter = Infinity;\n\
      var cur = posFromMouse(cm, e);\n\
      if (cur) doSelect(cur);\n\
      e_preventDefault(e);\n\
      focusInput(cm);\n\
      off(document, \"mousemove\", move);\n\
      off(document, \"mouseup\", up);\n\
    }\n\
\n\
    var move = operation(cm, function(e) {\n\
      e_preventDefault(e);\n\
      if (!ie && !e_button(e)) done(e);\n\
      else extend(e);\n\
    });\n\
    var up = operation(cm, done);\n\
    on(document, \"mousemove\", move);\n\
    on(document, \"mouseup\", up);\n\
  }\n\
\n\
  function onDrop(e) {\n\
    var cm = this;\n\
    if (cm.options.onDragEvent && cm.options.onDragEvent(cm, addStop(e))) return;\n\
    e_preventDefault(e);\n\
    var pos = posFromMouse(cm, e, true), files = e.dataTransfer.files;\n\
    if (!pos || cm.options.readOnly) return;\n\
    if (files && files.length && window.FileReader && window.File) {\n\
      var n = files.length, text = Array(n), read = 0;\n\
      var loadFile = function(file, i) {\n\
        var reader = new FileReader;\n\
        reader.onload = function() {\n\
          text[i] = reader.result;\n\
          if (++read == n) {\n\
            pos = clipPos(cm.view.doc, pos);\n\
            operation(cm, function() {\n\
              var end = replaceRange(cm, text.join(\"\"), pos, pos);\n\
              setSelectionUser(cm, pos, end);\n\
            })();\n\
          }\n\
        };\n\
        reader.readAsText(file);\n\
      };\n\
      for (var i = 0; i < n; ++i) loadFile(files[i], i);\n\
    } else {\n\
      // Don't do a replace if the drop happened inside of the selected text.\n\
      if (cm.view.draggingText && !(posLess(pos, cm.view.sel.from) || posLess(cm.view.sel.to, pos))) return;\n\
      try {\n\
        var text = e.dataTransfer.getData(\"Text\");\n\
        if (text) {\n\
          compoundChange(cm, function() {\n\
            var curFrom = cm.view.sel.from, curTo = cm.view.sel.to;\n\
            setSelectionUser(cm, pos, pos);\n\
            if (cm.view.draggingText) replaceRange(cm, \"\", curFrom, curTo);\n\
            cm.replaceSelection(text);\n\
            focusInput(cm);\n\
          });\n\
        }\n\
      }\n\
      catch(e){}\n\
    }\n\
  }\n\
\n\
  function clickInGutter(e) {\n\
    var cm = this, display = cm.display;\n\
    try { var mX = e.clientX, mY = e.clientY; }\n\
    catch(e) { return false; }\n\
    \n\
    if (mX >= Math.floor(display.gutters.getBoundingClientRect().right)) return false;\n\
    e_preventDefault(e);\n\
    if (!hasHandler(cm, \"gutterClick\")) return true;\n\
\n\
    var lineBox = display.lineDiv.getBoundingClientRect();\n\
    if (mY > lineBox.bottom) return true;\n\
    mY -= lineBox.top - display.viewOffset;\n\
\n\
    for (var i = 0; i < cm.options.gutters.length; ++i) {\n\
      var g = display.gutters.childNodes[i];\n\
      if (g && g.getBoundingClientRect().right >= mX) {\n\
        var line = lineAtHeight(cm.view.doc, mY);\n\
        var gutter = cm.options.gutters[i];\n\
        signalLater(cm, cm, \"gutterClick\", cm, line, gutter, e);\n\
        break;\n\
      }\n\
    }\n\
    return true;\n\
  }\n\
\n\
  function onDragStart(cm, e) {\n\
    var txt = cm.getSelection();\n\
    e.dataTransfer.setData(\"Text\", txt);\n\
\n\
    // Use dummy image instead of default browsers image.\n\
    if (e.dataTransfer.setDragImage)\n\
      e.dataTransfer.setDragImage(elt('img'), 0, 0);\n\
  }\n\
\n\
  function doHandleBinding(cm, bound, dropShift) {\n\
    if (typeof bound == \"string\") {\n\
      bound = commands[bound];\n\
      if (!bound) return false;\n\
    }\n\
    var view = cm.view, prevShift = view.sel.shift;\n\
    try {\n\
      if (cm.options.readOnly) view.suppressEdits = true;\n\
      if (dropShift) view.sel.shift = null;\n\
      bound(cm);\n\
    } catch(e) {\n\
      if (e != Pass) throw e;\n\
      return false;\n\
    } finally {\n\
      view.sel.shift = prevShift;\n\
      view.suppressEdits = false;\n\
    }\n\
    return true;\n\
  }\n\
\n\
  var maybeTransition;\n\
  function handleKeyBinding(cm, e) {\n\
    // Handle auto keymap transitions\n\
    var startMap = getKeyMap(cm.options.keyMap), next = startMap.auto;\n\
    clearTimeout(maybeTransition);\n\
    if (next && !isModifierKey(e)) maybeTransition = setTimeout(function() {\n\
      if (getKeyMap(cm.options.keyMap) == startMap)\n\
        cm.options.keyMap = (next.call ? next.call(null, cm) : next);\n\
    }, 50);\n\
\n\
    var name = keyNames[e_prop(e, \"keyCode\")], handled = false;\n\
    var flipCtrlCmd = opera && mac;\n\
    if (name == null || e.altGraphKey) return false;\n\
    if (e_prop(e, \"altKey\")) name = \"Alt-\" + name;\n\
    if (e_prop(e, flipCtrlCmd ? \"metaKey\" : \"ctrlKey\")) name = \"Ctrl-\" + name;\n\
    if (e_prop(e, flipCtrlCmd ? \"ctrlKey\" : \"metaKey\")) name = \"Cmd-\" + name;\n\
\n\
    var stopped = false;\n\
    function stop() { stopped = true; }\n\
\n\
    if (e_prop(e, \"shiftKey\")) {\n\
      handled = lookupKey(\"Shift-\" + name, cm.options.extraKeys, cm.options.keyMap,\n\
                          function(b) {return doHandleBinding(cm, b, true);}, stop)\n\
        || lookupKey(name, cm.options.extraKeys, cm.options.keyMap, function(b) {\n\
          if (typeof b == \"string\" && /^go[A-Z]/.test(b)) return doHandleBinding(cm, b);\n\
        }, stop);\n\
    } else {\n\
      handled = lookupKey(name, cm.options.extraKeys, cm.options.keyMap,\n\
                          function(b) { return doHandleBinding(cm, b); }, stop);\n\
    }\n\
    if (stopped) handled = false;\n\
    if (handled) {\n\
      e_preventDefault(e);\n\
      restartBlink(cm);\n\
      if (ie) { e.oldKeyCode = e.keyCode; e.keyCode = 0; }\n\
    }\n\
    return handled;\n\
  }\n\
\n\
  function handleCharBinding(cm, e, ch) {\n\
    var handled = lookupKey(\"'\" + ch + \"'\", cm.options.extraKeys, cm.options.keyMap,\n\
                            function(b) { return doHandleBinding(cm, b, true); });\n\
    if (handled) {\n\
      e_preventDefault(e);\n\
      restartBlink(cm);\n\
    }\n\
    return handled;\n\
  }\n\
\n\
  var lastStoppedKey = null;\n\
  function onKeyDown(e) {\n\
    var cm = this;\n\
    if (!cm.view.focused) onFocus(cm);\n\
    if (ie && e.keyCode == 27) { e.returnValue = false; }\n\
    if (cm.display.pollingFast) { if (readInput(cm)) cm.display.pollingFast = false; }\n\
    if (cm.options.onKeyEvent && cm.options.onKeyEvent(cm, addStop(e))) return;\n\
    var code = e_prop(e, \"keyCode\");\n\
    // IE does strange things with escape.\n\
    setShift(cm.view, code == 16 || e_prop(e, \"shiftKey\"));\n\
    // First give onKeyEvent option a chance to handle this.\n\
    var handled = handleKeyBinding(cm, e);\n\
    if (opera) {\n\
      lastStoppedKey = handled ? code : null;\n\
      // Opera has no cut event... we try to at least catch the key combo\n\
      if (!handled && code == 88 && e_prop(e, mac ? \"metaKey\" : \"ctrlKey\"))\n\
        cm.replaceSelection(\"\");\n\
    }\n\
  }\n\
\n\
  function onKeyPress(e) {\n\
    var cm = this;\n\
    if (cm.display.pollingFast) readInput(cm);\n\
    if (cm.options.onKeyEvent && cm.options.onKeyEvent(cm, addStop(e))) return;\n\
    var keyCode = e_prop(e, \"keyCode\"), charCode = e_prop(e, \"charCode\");\n\
    if (opera && keyCode == lastStoppedKey) {lastStoppedKey = null; e_preventDefault(e); return;}\n\
    if (((opera && (!e.which || e.which < 10)) || khtml) && handleKeyBinding(cm, e)) return;\n\
    var ch = String.fromCharCode(charCode == null ? keyCode : charCode);\n\
    if (this.options.electricChars && this.view.doc.mode.electricChars &&\n\
        this.options.smartIndent && !this.options.readOnly &&\n\
        this.view.doc.mode.electricChars.indexOf(ch) > -1)\n\
      setTimeout(operation(cm, function() {indentLine(cm, cm.view.sel.to.line, \"smart\");}), 75);\n\
    if (handleCharBinding(cm, e, ch)) return;\n\
    fastPoll(cm);\n\
  }\n\
\n\
  function onFocus(cm) {\n\
    if (cm.options.readOnly == \"nocursor\") return;\n\
    if (!cm.view.focused) {\n\
      signal(cm, \"focus\", cm);\n\
      cm.view.focused = true;\n\
      if (cm.display.scroller.className.search(/\\bCodeMirror-focused\\b/) == -1)\n\
        cm.display.scroller.className += \" CodeMirror-focused\";\n\
    }\n\
    slowPoll(cm);\n\
    restartBlink(cm);\n\
  }\n\
  function onBlur(cm) {\n\
    if (cm.view.focused) {\n\
      signal(cm, \"blur\", cm);\n\
      cm.view.focused = false;\n\
      cm.display.scroller.className = cm.display.scroller.className.replace(\" CodeMirror-focused\", \"\");\n\
    }\n\
    clearInterval(cm.display.blinker);\n\
    setTimeout(function() {if (!cm.view.focused) cm.view.sel.shift = null;}, 150);\n\
  }\n\
\n\
  var detectingSelectAll;\n\
  function onContextMenu(cm, e) {\n\
    var display = cm.display, sel = cm.view.sel;\n\
    var pos = posFromMouse(cm, e), scrollPos = display.scroller.scrollTop;\n\
    if (!pos || opera) return; // Opera is difficult.\n\
    if (posEq(sel.from, sel.to) || posLess(pos, sel.from) || !posLess(pos, sel.to))\n\
      operation(cm, setSelection)(cm, pos, pos);\n\
\n\
    var oldCSS = display.input.style.cssText;\n\
    display.inputDiv.style.position = \"absolute\";\n\
    display.input.style.cssText = \"position: fixed; width: 30px; height: 30px; top: \" + (e.clientY - 5) +\n\
      \"px; left: \" + (e.clientX - 5) + \"px; z-index: 1000; background: white; outline: none;\" +\n\
      \"border-width: 0; outline: none; overflow: hidden; opacity: .05; filter: alpha(opacity=5);\";\n\
    focusInput(cm);\n\
    resetInput(cm, true);\n\
    // Adds \"Select all\" to context menu in FF\n\
    if (posEq(sel.from, sel.to)) display.input.value = display.prevInput = \" \";\n\
\n\
    function rehide() {\n\
      display.inputDiv.style.position = \"relative\";\n\
      display.input.style.cssText = oldCSS;\n\
      if (ie_lt9) display.scrollbarV.scrollTop = display.scroller.scrollTop = scrollPos;\n\
      slowPoll(cm);\n\
\n\
      // Try to detect the user choosing select-all \n\
      if (display.input.selectionStart != null) {\n\
        clearTimeout(detectingSelectAll);\n\
        var extval = display.input.value = \" \" + (posEq(sel.from, sel.to) ? \"\" : display.input.value), i = 0;\n\
        display.prevInput = \" \";\n\
        display.input.selectionStart = 1; display.input.selectionEnd = extval.length;\n\
        detectingSelectAll = setTimeout(function poll(){\n\
          if (display.prevInput == \" \" && display.input.selectionStart == 0)\n\
            operation(cm, commands.selectAll)(cm);\n\
          else if (i++ < 10) detectingSelectAll = setTimeout(poll, 500);\n\
          else resetInput(cm);\n\
        }, 200);\n\
      }\n\
    }\n\
\n\
    if (gecko) {\n\
      e_stop(e);\n\
      on(window, \"mouseup\", function mouseup() {\n\
        off(window, \"mouseup\", mouseup);\n\
        setTimeout(rehide, 20);\n\
      });\n\
    } else {\n\
      setTimeout(rehide, 50);\n\
    }\n\
  }\n\
\n\
  // UPDATING\n\
\n\
  // Replace the range from from to to by the strings in newText.\n\
  // Afterwards, set the selection to selFrom, selTo.\n\
  function updateDoc(cm, from, to, newText, selFrom, selTo) {\n\
    var view = cm.view, doc = view.doc;\n\
    if (view.suppressEdits) return;\n\
    var old = [];\n\
    doc.iter(from.line, to.line + 1, function(line) {\n\
      old.push(newHL(line.text, line.markedSpans));\n\
    });\n\
    if (doc.history) {\n\
      doc.history.addChange(from.line, newText.length, old);\n\
      while (doc.history.done.length > cm.options.undoDepth) doc.history.done.shift();\n\
    }\n\
    var lines = updateMarkedSpans(hlSpans(old[0]), hlSpans(lst(old)), from.ch, to.ch, newText);\n\
    updateDocNoUndo(cm, from, to, lines, selFrom, selTo);\n\
  }\n\
  function unredoHelper(cm, from, to) {\n\
    var doc = cm.view.doc;\n\
    if (!from.length) return;\n\
    var set = from.pop(), out = [];\n\
    for (var i = set.length - 1; i >= 0; i -= 1) {\n\
      var change = set[i];\n\
      var replaced = [], end = change.start + change.added;\n\
      doc.iter(change.start, end, function(line) { replaced.push(newHL(line.text, line.markedSpans)); });\n\
      out.push({start: change.start, added: change.old.length, old: replaced});\n\
      var pos = {line: change.start + change.old.length - 1,\n\
                 ch: editEnd(hlText(lst(replaced)), hlText(lst(change.old)))};\n\
      updateDocNoUndo(cm, {line: change.start, ch: 0}, {line: end - 1, ch: getLine(doc, end-1).text.length},\n\
                      change.old, pos, pos);\n\
    }\n\
    to.push(out);\n\
  }\n\
  function undo(cm) {\n\
    var hist = cm.view.doc.history;\n\
    unredoHelper(cm, hist.done, hist.undone);\n\
  }\n\
  function redo(cm) {\n\
    var hist = cm.view.doc.history;\n\
    unredoHelper(cm, hist.undone, hist.done);\n\
  }\n\
\n\
  function updateDocNoUndo(cm, from, to, lines, selFrom, selTo) {\n\
    var view = cm.view, doc = view.doc, display = cm.display;\n\
    if (view.suppressEdits) return;\n\
    var recomputeMaxLength = false, maxLineLength = view.maxLine.text.length;\n\
    if (!cm.options.lineWrapping)\n\
      doc.iter(from.line, to.line + 1, function(line) {\n\
        if (!line.hidden && line.text.length == maxLineLength) {recomputeMaxLength = true; return true;}\n\
      });\n\
\n\
    var nlines = to.line - from.line, firstLine = getLine(doc, from.line), lastLine = getLine(doc, to.line);\n\
    var lastHL = lst(lines), th = textHeight(display);\n\
\n\
    // First adjust the line structure\n\
    if (from.ch == 0 && to.ch == 0 && hlText(lastHL) == \"\") {\n\
      // This is a whole-line replace. Treated specially to make\n\
      // sure line objects move the way they are supposed to.\n\
      var added = [], prevLine = null;\n\
      for (var i = 0, e = lines.length - 1; i < e; ++i)\n\
        added.push(new Line(hlText(lines[i]), hlSpans(lines[i]), th));\n\
      lastLine.update(lastLine.text, hlSpans(lastHL), cm);\n\
      if (nlines) doc.remove(from.line, nlines, cm);\n\
      if (added.length) doc.insert(from.line, added);\n\
    } else if (firstLine == lastLine) {\n\
      if (lines.length == 1) {\n\
        firstLine.update(firstLine.text.slice(0, from.ch) + hlText(lines[0]) + firstLine.text.slice(to.ch),\n\
                         hlSpans(lines[0]), cm);\n\
      } else {\n\
        for (var added = [], i = 1, e = lines.length - 1; i < e; ++i)\n\
          added.push(new Line(hlText(lines[i]), hlSpans(lines[i]), th));\n\
        added.push(new Line(hlText(lastHL) + firstLine.text.slice(to.ch), hlSpans(lastHL), th));\n\
        firstLine.update(firstLine.text.slice(0, from.ch) + hlText(lines[0]), hlSpans(lines[0]), cm);\n\
        doc.insert(from.line + 1, added);\n\
      }\n\
    } else if (lines.length == 1) {\n\
      firstLine.update(firstLine.text.slice(0, from.ch) + hlText(lines[0]) + lastLine.text.slice(to.ch),\n\
                       hlSpans(lines[0]), cm);\n\
      doc.remove(from.line + 1, nlines, cm);\n\
    } else {\n\
      var added = [];\n\
      firstLine.update(firstLine.text.slice(0, from.ch) + hlText(lines[0]), hlSpans(lines[0]), cm);\n\
      lastLine.update(hlText(lastHL) + lastLine.text.slice(to.ch), hlSpans(lastHL), cm);\n\
      for (var i = 1, e = lines.length - 1; i < e; ++i)\n\
        added.push(new Line(hlText(lines[i]), hlSpans(lines[i]), th));\n\
      if (nlines > 1) doc.remove(from.line + 1, nlines - 1, cm);\n\
      doc.insert(from.line + 1, added);\n\
    }\n\
\n\
    if (cm.options.lineWrapping) {\n\
      var perLine = Math.max(5, display.scroller.clientWidth / charWidth(display) - 3);\n\
      doc.iter(from.line, from.line + lines.length, function(line) {\n\
        if (line.hidden) return;\n\
        var guess = (Math.ceil(line.text.length / perLine) || 1) * th;\n\
        if (guess != line.height) updateLineHeight(line, guess);\n\
      });\n\
    } else {\n\
      doc.iter(from.line, from.line + lines.length, function(line) {\n\
        var l = line.text;\n\
        if (!line.hidden && l.length > maxLineLength) {\n\
          view.maxLine = line; maxLineLength = l.length; view.maxLineChanged = true;\n\
          recomputeMaxLength = false;\n\
        }\n\
      });\n\
      if (recomputeMaxLength) cm.curOp.updateMaxLine = true;\n\
    }\n\
\n\
    // Adjust frontier, schedule worker\n\
    doc.frontier = Math.min(doc.frontier, from.line);\n\
    startWorker(cm, 400);\n\
\n\
    var lendiff = lines.length - nlines - 1;\n\
    // Remember that these lines changed, for updating the display\n\
    regChange(cm, from.line, to.line + 1, lendiff);\n\
    if (hasHandler(cm, \"change\")) {\n\
      // Normalize lines to contain only strings, since that's what\n\
      // the change event handler expects\n\
      for (var i = 0; i < lines.length; ++i)\n\
        if (typeof lines[i] != \"string\") lines[i] = lines[i].text;\n\
      var changeObj = {from: from, to: to, text: lines};\n\
      if (cm.curOp.textChanged) {\n\
        for (var cur = cm.curOp.textChanged; cur.next; cur = cur.next) {}\n\
        cur.next = changeObj;\n\
      } else cm.curOp.textChanged = changeObj;\n\
    }\n\
\n\
    // Update the selection\n\
    setSelection(cm, clipPos(doc, selFrom), clipPos(doc, selTo), true);\n\
  }\n\
\n\
  function replaceRange(cm, code, from, to) {\n\
    if (!to) to = from;\n\
    if (posLess(to, from)) { var tmp = to; to = from; from = tmp; }\n\
    code = splitLines(code);\n\
    function adjustPos(pos) {\n\
      if (posLess(pos, from)) return pos;\n\
      if (!posLess(to, pos)) return end;\n\
      var line = pos.line + code.length - (to.line - from.line) - 1;\n\
      var ch = pos.ch;\n\
      if (pos.line == to.line)\n\
        ch += lst(code).length - (to.ch - (to.line == from.line ? from.ch : 0));\n\
      return {line: line, ch: ch};\n\
    }\n\
    var end;\n\
    replaceRange1(cm, code, from, to, function(end1) {\n\
      end = end1;\n\
      return {from: adjustPos(cm.view.sel.from), to: adjustPos(cm.view.sel.to)};\n\
    });\n\
    return end;\n\
  }\n\
  function replaceRange1(cm, code, from, to, computeSel) {\n\
    var endch = code.length == 1 ? code[0].length + from.ch : lst(code).length;\n\
    var newSel = computeSel({line: from.line + code.length - 1, ch: endch});\n\
    updateDoc(cm, from, to, code, newSel.from, newSel.to);\n\
  }\n\
\n\
  // SELECTION\n\
\n\
  function posEq(a, b) {return a.line == b.line && a.ch == b.ch;}\n\
  function posLess(a, b) {return a.line < b.line || (a.line == b.line && a.ch < b.ch);}\n\
  function copyPos(x) {return {line: x.line, ch: x.ch};}\n\
\n\
  function clipLine(doc, n) {return Math.max(0, Math.min(n, doc.size-1));}\n\
  function clipPos(doc, pos) {\n\
    if (pos.line < 0) return {line: 0, ch: 0};\n\
    if (pos.line >= doc.size) return {line: doc.size-1, ch: getLine(doc, doc.size-1).text.length};\n\
    var ch = pos.ch, linelen = getLine(doc, pos.line).text.length;\n\
    if (ch == null || ch > linelen) return {line: pos.line, ch: linelen};\n\
    else if (ch < 0) return {line: pos.line, ch: 0};\n\
    else return pos;\n\
  }\n\
  function isLine(doc, l) {return l >= 0 && l < doc.size;}\n\
\n\
  function setShift(view, val) {\n\
    if (val) view.sel.shift = view.sel.shift || selHead(view);\n\
    else view.sel.shift = null;\n\
  }\n\
  function setSelectionUser(cm, from, to) {\n\
    var view = cm.view, sh = view.sel.shift;\n\
    if (sh) {\n\
      sh = clipPos(view.doc, sh);\n\
      if (posLess(sh, from)) from = sh;\n\
      else if (posLess(to, sh)) to = sh;\n\
    }\n\
    setSelection(cm, from, to);\n\
    cm.curOp.userSelChange = true;\n\
  }\n\
\n\
  // Update the selection. Last two args are only used by\n\
  // updateDoc, since they have to be expressed in the line\n\
  // numbers before the update.\n\
  function setSelection(cm, from, to, isChange) {\n\
    cm.curOp.updateInput = true;\n\
    var sel = cm.view.sel, doc = cm.view.doc;\n\
    cm.view.goalColumn = null;\n\
    if (posEq(sel.from, from) && posEq(sel.to, to)) return;\n\
    if (posLess(to, from)) {var tmp = to; to = from; from = tmp;}\n\
\n\
    // Skip over hidden lines.\n\
    if (isChange || from.line != sel.from.line) {\n\
      var from1 = skipHidden(doc, from, sel.from.line, sel.from.ch, cm);\n\
      // If there is no non-hidden line left, force visibility on current line\n\
      if (!from1) cm.unfoldLines(getLine(doc, from.line).hidden.id);\n\
      else from = from1;\n\
    }\n\
    if (isChange || to.line != sel.to.line) to = skipHidden(doc, to, sel.to.line, sel.to.ch, cm);\n\
\n\
    if (posEq(from, to)) sel.inverted = false;\n\
    else if (posEq(from, sel.to)) sel.inverted = false;\n\
    else if (posEq(to, sel.from)) sel.inverted = true;\n\
\n\
    if (cm.options.autoClearEmptyLines && posEq(sel.from, sel.to)) {\n\
      var head = selHead(cm.view);\n\
      if (head.line != sel.from.line && sel.from.line < doc.size) {\n\
        var oldLine = getLine(doc, sel.from.line);\n\
        if (/^\\s+$/.test(oldLine.text))\n\
          setTimeout(operation(cm, function() {\n\
            if (oldLine.parent && /^\\s+$/.test(oldLine.text)) {\n\
              var no = lineNo(oldLine);\n\
              replaceRange(cm, \"\", {line: no, ch: 0}, {line: no, ch: oldLine.text.length});\n\
            }\n\
          }, 10));\n\
      }\n\
    }\n\
\n\
    sel.from = from; sel.to = to;\n\
    cm.curOp.selectionChanged = true;\n\
  }\n\
\n\
  function skipHidden(doc, pos, oldLine, oldCh, allowUnfold) {\n\
    function getNonHidden(dir) {\n\
      var lNo = pos.line + dir, end = dir == 1 ? doc.size : -1;\n\
      while (lNo != end) {\n\
        var line = getLine(doc, lNo);\n\
        if (!line.hidden) {\n\
          var ch = pos.ch;\n\
          if (toEnd || ch > oldCh || ch > line.text.length) ch = line.text.length;\n\
          return {line: lNo, ch: ch};\n\
        }\n\
        lNo += dir;\n\
      }\n\
    }\n\
    var line = getLine(doc, pos.line);\n\
    while (allowUnfold && line.hidden && line.hidden.handle.unfoldOnEnter)\n\
      allowUnfold.unfoldLines(line.hidden.handle);\n\
    if (!line.hidden) return pos;\n\
    var toEnd = pos.ch == line.text.length && pos.ch != oldCh;\n\
    if (pos.line >= oldLine) return getNonHidden(1) || getNonHidden(-1);\n\
    else return getNonHidden(-1) || getNonHidden(1);\n\
  }\n\
\n\
  // SCROLLING\n\
\n\
  function scrollCursorIntoView(cm) {\n\
    var view = cm.view, coords = cursorCoords(cm, selHead(view));\n\
    scrollIntoView(cm.display, coords.left, coords.top, coords.left, coords.bottom);\n\
    if (!view.focused) return;\n\
    var display = cm.display, box = display.sizer.getBoundingClientRect(), doScroll = null;\n\
    if (coords.top + box.top < 0) doScroll = true;\n\
    else if (coords.bottom + box.top > (window.innerHeight || document.documentElement.clientHeight)) doScroll = false;\n\
    if (doScroll != null) {\n\
      var hidden = display.cursor.style.display == \"none\";\n\
      if (hidden) {\n\
        display.cursor.style.display = \"\";\n\
        display.cursor.style.left = coords.left + \"px\";\n\
        display.cursor.style.top = (coords.top - display.viewOffset) + \"px\";\n\
      }\n\
      display.cursor.scrollIntoView(doScroll);\n\
      if (hidden) display.cursor.style.display = \"none\";\n\
    }\n\
  }\n\
\n\
  function scrollIntoView(display, x1, y1, x2, y2) {\n\
    var scrollPos = calculateScrollPos(display, x1, y1, x2, y2);\n\
    if (scrollPos.scrollLeft != null) {display.scrollbarH.scrollLeft = display.scroller.scrollLeft = scrollPos.scrollLeft;}\n\
    if (scrollPos.scrollTop != null) {display.scrollbarV.scrollTop = display.scroller.scrollTop = scrollPos.scrollTop;}\n\
  }\n\
\n\
  function calculateScrollPos(display, x1, y1, x2, y2) {\n\
    var pt = paddingTop(display);\n\
    y1 += pt; y2 += pt;\n\
    var screen = display.scroller.clientHeight - scrollerCutOff, screentop = display.scroller.scrollTop, result = {};\n\
    var docBottom = display.scroller.scrollHeight - scrollerCutOff;\n\
    var atTop = y1 < pt + 10, atBottom = y2 + pt > docBottom - 10;\n\
    if (y1 < screentop) result.scrollTop = atTop ? 0 : Math.max(0, y1);\n\
    else if (y2 > screentop + screen) result.scrollTop = (atBottom ? docBottom : y2 - screen);\n\
\n\
    var screenw = display.scroller.clientWidth - scrollerCutOff, screenleft = display.scroller.scrollLeft;\n\
    x1 += display.gutters.offsetWidth; x2 += display.gutters.offsetWidth;\n\
    var gutterw = display.gutters.offsetWidth;\n\
    var atLeft = x1 < gutterw + 10;\n\
    if (x1 < screenleft + gutterw || atLeft) {\n\
      if (atLeft) x1 = 0;\n\
      result.scrollLeft = Math.max(0, x1 - 10 - gutterw);\n\
    } else if (x2 > screenw + screenleft - 3) {\n\
      result.scrollLeft = x2 + 10 - screenw;\n\
    }\n\
    return result;\n\
  }\n\
\n\
  // API UTILITIES\n\
\n\
  function indentLine(cm, n, how) {\n\
    var doc = cm.view.doc;\n\
    if (!how) how = \"add\";\n\
    if (how == \"smart\") {\n\
      if (!doc.mode.indent) how = \"prev\";\n\
      else var state = getStateBefore(doc, n);\n\
    }\n\
\n\
    var line = getLine(doc, n), curSpace = line.indentation(doc.tabSize),\n\
    curSpaceString = line.text.match(/^\\s*/)[0], indentation;\n\
    if (how == \"smart\") {\n\
      indentation = doc.mode.indent(state, line.text.slice(curSpaceString.length), line.text);\n\
      if (indentation == Pass) how = \"prev\";\n\
    }\n\
    if (how == \"prev\") {\n\
      if (n) indentation = getLine(doc, n-1).indentation(doc.tabSize);\n\
      else indentation = 0;\n\
    }\n\
    else if (how == \"add\") indentation = curSpace + cm.options.indentUnit;\n\
    else if (how == \"subtract\") indentation = curSpace - cm.options.indentUnit;\n\
    indentation = Math.max(0, indentation);\n\
    var diff = indentation - curSpace;\n\
\n\
    var indentString = \"\", pos = 0;\n\
    if (cm.options.indentWithTabs)\n\
      for (var i = Math.floor(indentation / doc.tabSize); i; --i) {pos += doc.tabSize; indentString += \"\\t\";}\n\
    if (pos < indentation) indentString += spaceStr(indentation - pos);\n\
\n\
    if (indentString != curSpaceString)\n\
      replaceRange(cm, indentString, {line: n, ch: 0}, {line: n, ch: curSpaceString.length});\n\
  }\n\
\n\
  function changeLine(cm, handle, op) {\n\
    var no = handle, line = handle, doc = cm.view.doc;\n\
    if (typeof handle == \"number\") line = getLine(doc, clipLine(doc, handle));\n\
    else no = lineNo(handle);\n\
    if (no == null) return null;\n\
    if (op(line, no)) regChange(cm, no, no + 1);\n\
    else return null;\n\
    return line;\n\
  }\n\
\n\
  function findPosH(cm, dir, unit, visually) {\n\
    var doc = cm.view.doc, end = selHead(cm.view), line = end.line, ch = end.ch;\n\
    var lineObj = getLine(doc, line);\n\
    function findNextLine() {\n\
      for (var l = line + dir, e = dir < 0 ? -1 : doc.size; l != e; l += dir) {\n\
        var lo = getLine(doc, l);\n\
        if (!lo.hidden || !lo.hidden.handle.unfoldOnEneter) { line = l; lineObj = lo; return true; }\n\
      }\n\
    }\n\
    function moveOnce(boundToLine) {\n\
      var next = (visually ? moveVisually : moveLogically)(lineObj, ch, dir, true);\n\
      if (next == null) {\n\
        if (!boundToLine && findNextLine()) {\n\
          if (visually) ch = (dir < 0 ? lineRight : lineLeft)(lineObj);\n\
          else ch = dir < 0 ? lineObj.text.length : 0;\n\
        } else return false;\n\
      } else ch = next;\n\
      return true;\n\
    }\n\
    if (unit == \"char\") moveOnce();\n\
    else if (unit == \"column\") moveOnce(true);\n\
    else if (unit == \"word\") {\n\
      var sawWord = false;\n\
      for (;;) {\n\
        if (dir < 0) if (!moveOnce()) break;\n\
        if (isWordChar(lineObj.text.charAt(ch))) sawWord = true;\n\
        else if (sawWord) {if (dir < 0) {dir = 1; moveOnce();} break;}\n\
        if (dir > 0) if (!moveOnce()) break;\n\
      }\n\
    }\n\
    return {line: line, ch: ch};\n\
  }\n\
\n\
  function findWordAt(line, pos) {\n\
    var start = pos.ch, end = pos.ch;\n\
    if (line) {\n\
      if (pos.after === false || end == line.length) --start; else ++end;\n\
      var startChar = line.charAt(start);\n\
      var check = isWordChar(startChar) ? isWordChar :\n\
        /\\s/.test(startChar) ? function(ch) {return /\\s/.test(ch);} :\n\
      function(ch) {return !/\\s/.test(ch) && !isWordChar(ch);};\n\
      while (start > 0 && check(line.charAt(start - 1))) --start;\n\
      while (end < line.length && check(line.charAt(end))) ++end;\n\
    }\n\
    return {from: {line: pos.line, ch: start}, to: {line: pos.line, ch: end}};\n\
  }\n\
\n\
  function selectLine(cm, line) {\n\
    setSelectionUser(cm, {line: line, ch: 0}, clipPos(cm.view.doc, {line: line + 1, ch: 0}));\n\
  }\n\
\n\
  // PROTOTYPE\n\
\n\
  // The publicly visible API. Note that operation(null, f) means\n\
  // 'wrap f in an operation, performed on its `this` parameter'\n\
\n\
  CodeMirror.prototype = {\n\
    getValue: function(lineSep) {\n\
      var text = [], doc = this.view.doc;\n\
      doc.iter(0, doc.size, function(line) { text.push(line.text); });\n\
      return text.join(lineSep || \"\\n\");\n\
    },\n\
\n\
    setValue: operation(null, function(code) {\n\
      var doc = this.view.doc, top = {line: 0, ch: 0}, lastLen = getLine(doc, doc.size-1).text.length;\n\
      updateDoc(this, top, {line: doc.size - 1, ch: lastLen}, splitLines(code), top, top);\n\
    }),\n\
\n\
    getSelection: function(lineSep) { return this.getRange(this.view.sel.from, this.view.sel.to, lineSep); },\n\
\n\
    replaceSelection: operation(null, function(code, collapse) {\n\
      var sel = this.view.sel;\n\
      replaceRange1(this, splitLines(code), sel.from, sel.to, function(end) {\n\
        if (collapse == \"end\") return {from: end, to: end};\n\
        else if (collapse == \"start\") return {from: sel.from, to: sel.from};\n\
        else return {from: sel.from, to: end};\n\
      });\n\
    }),\n\
\n\
    focus: function(){window.focus(); focusInput(this); onFocus(this); fastPoll(this);},\n\
\n\
    setOption: function(option, value) {\n\
      var options = this.options;\n\
      if (options[option] == value && option != \"mode\") return;\n\
      options[option] = value;\n\
      if (option == \"mode\" || option == \"indentUnit\") loadMode(this);\n\
      else if (option == \"readOnly\" && value == \"nocursor\") {onBlur(this); this.display.input.blur();}\n\
      else if (option == \"readOnly\" && !value) {resetInput(this, true);}\n\
      else if (option == \"theme\") themeChanged(this);\n\
      else if (option == \"lineWrapping\") operation(this, wrappingChanged)(this);\n\
      else if (option == \"tabSize\") {this.view.doc.tabSize = value; updateDisplay(this, true);}\n\
      else if (option == \"keyMap\") keyMapChanged(this);\n\
      else if (option == \"gutters\" || option == \"lineNumbers\") setGuttersForLineNumbers(this.options);\n\
      if (option == \"lineNumbers\" || option == \"gutters\" || option == \"firstLineNumber\" ||\n\
          option == \"theme\" || option == \"lineNumberFormatter\")\n\
        guttersChanged(this);\n\
      if (optionHandlers.hasOwnProperty(option))\n\
        optionHandlers[option](this, value);\n\
    },\n\
\n\
    getOption: function(option) {return this.options[option];},\n\
\n\
    getMode: function() {return this.view.doc.mode;},\n\
\n\
    undo: operation(null, function() {\n\
      var hist = this.view.doc.history;\n\
      unredoHelper(this, hist.done, hist.undone);\n\
    }),\n\
    redo: operation(null, function() {\n\
      var hist = this.view.doc.history;\n\
      unredoHelper(this, hist.undone, hist.done);\n\
    }),\n\
\n\
    indentLine: operation(null, function(n, dir) {\n\
      if (typeof dir != \"string\") {\n\
        if (dir == null) dir = this.options.smartIndent ? \"smart\" : \"prev\";\n\
        else dir = dir ? \"add\" : \"subtract\";\n\
      }\n\
      if (isLine(this.view.doc, n)) indentLine(this, n, dir);\n\
    }),\n\
\n\
    indentSelection: operation(null, function(how) {\n\
      var sel = this.view.sel;\n\
      if (posEq(sel.from, sel.to)) return indentLine(this, sel.from.line, how);\n\
      var e = sel.to.line - (sel.to.ch ? 0 : 1);\n\
      for (var i = sel.from.line; i <= e; ++i) indentLine(this, i, how);\n\
    }),\n\
\n\
    historySize: function() {\n\
      var hist = this.view.doc.history;\n\
      return {undo: hist.done.length, redo: hist.undone.length};\n\
    },\n\
\n\
    clearHistory: function() {this.view.doc.history = new History();},\n\
\n\
    getHistory: function() {\n\
      var hist = this.view.doc.history;\n\
      function cp(arr) {\n\
        for (var i = 0, nw = [], nwelt; i < arr.length; ++i) {\n\
          nw.push(nwelt = []);\n\
          for (var j = 0, elt = arr[i]; j < elt.length; ++j) {\n\
            var old = [], cur = elt[j];\n\
            nwelt.push({start: cur.start, added: cur.added, old: old});\n\
            for (var k = 0; k < cur.old.length; ++k) old.push(hlText(cur.old[k]));\n\
          }\n\
        }\n\
        return nw;\n\
      }\n\
      return {done: cp(hist.done), undone: cp(hist.undone)};\n\
    },\n\
\n\
    setHistory: function(histData) {\n\
      var hist = this.view.doc.history = new History();\n\
      hist.done = histData.done;\n\
      hist.undone = histData.undone;\n\
    },\n\
\n\
    getTokenAt: function(pos) {\n\
      var doc = this.view.doc;\n\
      pos = clipPos(doc, pos);\n\
      return getLine(doc, pos.line).getTokenAt(doc.mode, getStateBefore(doc, pos.line),\n\
                                               this.options.tabSize, pos.ch);\n\
    },\n\
\n\
    getStateAfter: function(line) {\n\
      var doc = this.view.doc;\n\
      line = clipLine(doc, line == null ? doc.size - 1: line);\n\
      return getStateBefore(doc, line + 1);\n\
    },\n\
\n\
    cursorCoords: function(start, mode) {\n\
      var pos, sel = this.view.sel;\n\
      if (start == null) start = sel.inverted;\n\
      if (typeof start == \"object\") pos = clipPos(this.view.doc, start);\n\
      else pos = start ? sel.from : sel.to;\n\
      return cursorCoords(this, pos, mode || \"page\");\n\
    },\n\
\n\
    charCoords: function(pos, mode) {\n\
      return charCoords(this, clipPos(this.view.doc, pos), mode || \"page\");\n\
    },\n\
\n\
    coordsChar: function(coords) {\n\
      var off = this.display.lineSpace.getBoundingClientRect();\n\
      return coordsChar(this, coords.left - off.left, coords.top - off.top);\n\
    },\n\
\n\
    markText: operation(null, function(from, to, className, options) {\n\
      var doc = this.view.doc;\n\
      from = clipPos(doc, from); to = clipPos(doc, to);\n\
      var marker = new TextMarker(this, \"range\", className);\n\
      if (options) for (var opt in options) if (options.hasOwnProperty(opt))\n\
        marker[opt] = options[opt];\n\
      var curLine = from.line;\n\
      doc.iter(curLine, to.line + 1, function(line) {\n\
        var span = {from: curLine == from.line ? from.ch : null,\n\
                    to: curLine == to.line ? to.ch : null,\n\
                    marker: marker};\n\
        (line.markedSpans || (line.markedSpans = [])).push(span);\n\
        marker.lines.push(line);\n\
        ++curLine;\n\
      });\n\
      regChange(this, from.line, to.line + 1);\n\
      return marker;\n\
    }),\n\
\n\
    setBookmark: function(pos) {\n\
      var doc = this.view.doc;\n\
      pos = clipPos(doc, pos);\n\
      var marker = new TextMarker(this, \"bookmark\"), line = getLine(doc, pos.line);\n\
      var span = {from: pos.ch, to: pos.ch, marker: marker};\n\
      (line.markedSpans || (line.markedSpans = [])).push(span);\n\
      marker.lines.push(line);\n\
      return marker;\n\
    },\n\
\n\
    findMarksAt: function(pos) {\n\
      var doc = this.view.doc;\n\
      pos = clipPos(doc, pos);\n\
      var markers = [], spans = getLine(doc, pos.line).markedSpans;\n\
      if (spans) for (var i = 0; i < spans.length; ++i) {\n\
        var span = spans[i];\n\
        if ((span.from == null || span.from <= pos.ch) &&\n\
            (span.to == null || span.to >= pos.ch))\n\
          markers.push(span.marker);\n\
      }\n\
      return markers;\n\
    },\n\
\n\
    setGutterMarker: operation(null, function(line, gutterID, value) {\n\
      return changeLine(this, line, function(line) {\n\
        var markers = line.gutterMarkers || (line.gutterMarkers = {});\n\
        markers[gutterID] = value;\n\
        if (!value && isEmpty(markers)) line.gutterMarkers = null;\n\
        return true;\n\
      });\n\
    }),\n\
\n\
    clearGutter: operation(null, function(gutterID) {\n\
      var i = 0, cm = this, doc = cm.view.doc;\n\
      doc.iter(0, doc.size, function(line) {\n\
        if (line.gutterMarkers && line.gutterMarkers[gutterID]) {\n\
          line.gutterMarkers[gutterID] = null;\n\
          regChange(cm, i, i + 1);\n\
          if (isEmpty(line.gutterMarkers)) line.gutterMarkers = null;\n\
        }\n\
        ++i;\n\
      });\n\
    }),\n\
\n\
    setLineClass: operation(null, function(handle, className, bgClassName) {\n\
      return changeLine(this, handle, function(line) {\n\
        if (line.className != className || line.bgClassName != bgClassName) {\n\
          line.className = className;\n\
          line.bgClassName = bgClassName;\n\
          return true;\n\
        }\n\
      });\n\
    }),\n\
\n\
    addLineWidget: operation(null, function addLineWidget(handle, node, options) {\n\
      var widget = options || {};\n\
      widget.node = node;\n\
      if (widget.noHScroll) this.display.alignWidgets = true;\n\
      changeLine(this, handle, function(line) {\n\
        (line.widgets || (line.widgets = [])).push(widget);\n\
        widget.line = line;\n\
        return true;\n\
      });\n\
      return widget;\n\
    }),\n\
\n\
    removeLineWidget: operation(null, function(widget) {\n\
      var ws = widget.line.widgets, no = lineNo(widget.line);\n\
      if (no == null) return;\n\
      for (var i = 0; i < ws.length; ++i) if (ws[i] == widget) ws.splice(i--, 1);\n\
      regChange(this, no, no + 1);\n\
    }),\n\
\n\
    foldLines: operation(null, function(from, to, unfoldOnEnter) {\n\
      if (typeof from != \"number\") from = lineNo(from);\n\
      if (typeof to != \"number\") to = lineNo(to);\n\
      if (from > to) return;\n\
      var lines = [], handle = {lines: lines, unfoldOnEnter: unfoldOnEnter}, cm = this, view = cm.view, doc = view.doc;\n\
      doc.iter(from, to, function(line) {\n\
        lines.push(line);\n\
        if (!line.hidden && line.text.length == cm.view.maxLine.text.length)\n\
          cm.curOp.updateMaxLine = true;\n\
        line.hidden = {handle: handle, prev: line.hidden};\n\
        updateLineHeight(line, 0);\n\
      });\n\
      var sel = view.sel, selFrom = sel.from, selTo = sel.to;\n\
      if (selFrom.line >= from && selFrom.line < to)\n\
        selFrom = skipHidden(doc, {line: selFrom.line, ch: 0}, selFrom.line, 0);\n\
      if (selTo.line >= from && selTo.line < to)\n\
        selTo = skipHidden(doc, {line: selTo.line, ch: 0}, selTo.line, 0);\n\
      if (selFrom != sel.from || selTo != sel.to) setSelection(this, selFrom, selTo);\n\
      regChange(cm, from, to);\n\
      return handle;\n\
    }),\n\
\n\
    unfoldLines: operation(null, function(handle) {\n\
      var from, to;\n\
      for (var i = 0; i < handle.lines.length; ++i) {\n\
        var line = handle.lines[i], hidden = line.hidden;\n\
        if (!line.parent) continue;\n\
        if (hidden && hidden.handle == handle) line.hidden = line.hidden.prev;\n\
        else for (var h = hidden; h; h = h.prev) if (h.prev && h.prev.handle == handle) h.prev = h.prev.prev;\n\
        if (hidden && !line.hidden) {\n\
          var no = lineNo(handle);\n\
          from = Math.min(from, no); to = Math.max(to, no);\n\
          updateLineHeight(line, textHeight(this.display));\n\
          if (line.text.length > this.view.maxLine.text.length) {\n\
            this.view.maxLine = line;\n\
            this.view.maxLineChanged = true;\n\
          }\n\
        }\n\
      }\n\
      if (from != null) {\n\
        regChange(this, from, to + 1);\n\
        signalLater(this, handle, \"unfold\");\n\
      }\n\
    }),\n\
\n\
    lineInfo: function(line) {\n\
      if (typeof line == \"number\") {\n\
        if (!isLine(this.view.doc, line)) return null;\n\
        var n = line;\n\
        line = getLine(this.view.doc, line);\n\
        if (!line) return null;\n\
      } else {\n\
        var n = lineNo(line);\n\
        if (n == null) return null;\n\
      }\n\
      return {line: n, handle: line, text: line.text, gutterMarkers: line.gutterMarkers,\n\
              lineClass: line.className, bgClass: line.bgClassName, widgets: line.widgets};\n\
    },\n\
\n\
    getViewport: function() { return {from: this.display.showingFrom, to: this.display.showingTo};},\n\
\n\
    addWidget: function(pos, node, scroll, vert, horiz) {\n\
      var display = this.display;\n\
      pos = cursorCoords(this, clipPos(this.view.doc, pos));\n\
      var top = pos.top, left = pos.left;\n\
      node.style.position = \"absolute\";\n\
      display.sizer.appendChild(node);\n\
      if (vert == \"over\") top = pos.top;\n\
      else if (vert == \"near\") {\n\
        var vspace = Math.max(display.wrapper.clientHeight, this.view.doc.height),\n\
        hspace = Math.max(display.sizer.clientWidth, display.lineSpace.clientWidth);\n\
        if (pos.bottom + node.offsetHeight > vspace && pos.top > node.offsetHeight)\n\
          top = pos.top - node.offsetHeight;\n\
        if (left + node.offsetWidth > hspace)\n\
          left = hspace - node.offsetWidth;\n\
      }\n\
      node.style.top = (top + paddingTop(display)) + \"px\";\n\
      node.style.left = node.style.right = \"\";\n\
      if (horiz == \"right\") {\n\
        left = display.sizer.clientWidth - node.offsetWidth;\n\
        node.style.right = \"0px\";\n\
      } else {\n\
        if (horiz == \"left\") left = 0;\n\
        else if (horiz == \"middle\") left = (display.sizer.clientWidth - node.offsetWidth) / 2;\n\
        node.style.left = left + \"px\";\n\
      }\n\
      if (scroll)\n\
        scrollIntoView(display, left, top, left + node.offsetWidth, top + node.offsetHeight);\n\
    },\n\
\n\
    lineCount: function() {return this.view.doc.size;},\n\
\n\
    clipPos: function(pos) {return clipPos(this.view.doc, pos);},\n\
\n\
    getCursor: function(start) {\n\
      var sel = this.view.sel;\n\
      if (start == null) start = sel.inverted;\n\
      return copyPos(start ? sel.from : sel.to);\n\
    },\n\
\n\
    somethingSelected: function() {return !posEq(this.view.sel.from, this.view.sel.to);},\n\
\n\
    setCursor: operation(null, function(line, ch, user) {\n\
      var pos = typeof line == \"number\" ? {line: line, ch: ch || 0} : line;\n\
      (user ? setSelectionUser : setSelection)(this, pos, pos);\n\
    }),\n\
\n\
    setSelection: operation(null, function(from, to, user) {\n\
      var doc = this.view.doc;\n\
      (user ? setSelectionUser : setSelection)(this, clipPos(doc, from), clipPos(doc, to || from));\n\
    }),\n\
\n\
    getLine: function(line) {var l = this.getLineHandle(line); return l && l.text;},\n\
\n\
    getLineHandle: function(line) {\n\
      var doc = this.view.doc;\n\
      if (isLine(doc, line)) return getLine(doc, line);\n\
    },\n\
\n\
    getLineNumber: function(line) {return lineNo(line);},\n\
\n\
    setLine: operation(null, function(line, text) {\n\
      if (isLine(this.view.doc, line))\n\
        replaceRange(this, text, {line: line, ch: 0}, {line: line, ch: getLine(this.view.doc, line).text.length});\n\
    }),\n\
\n\
    removeLine: operation(null, function(line) {\n\
      if (isLine(this.view.doc, line))\n\
        replaceRange(this, \"\", {line: line, ch: 0}, clipPos(this.view.doc, {line: line+1, ch: 0}));\n\
    }),\n\
\n\
    replaceRange: operation(null, function(code, from, to) {\n\
      var doc = this.view.doc;\n\
      from = clipPos(doc, from);\n\
      to = to ? clipPos(doc, to) : from;\n\
      return replaceRange(this, code, from, to);\n\
    }),\n\
\n\
    getRange: function(from, to, lineSep) {\n\
      var doc = this.view.doc;\n\
      from = clipPos(doc, from); to = clipPos(doc, to);\n\
      var l1 = from.line, l2 = to.line;\n\
      if (l1 == l2) return getLine(doc, l1).text.slice(from.ch, to.ch);\n\
      var code = [getLine(doc, l1).text.slice(from.ch)];\n\
      doc.iter(l1 + 1, l2, function(line) { code.push(line.text); });\n\
      code.push(getLine(doc, l2).text.slice(0, to.ch));\n\
      return code.join(lineSep || \"\\n\");\n\
    },\n\
\n\
    triggerOnKeyDown: operation(null, onKeyDown),\n\
\n\
    execCommand: function(cmd) {return commands[cmd](this);},\n\
\n\
    // Stuff used by commands, probably not much use to outside code.\n\
    moveH: operation(null, function(dir, unit) {\n\
      var sel = this.view.sel, pos = dir < 0 ? sel.from : sel.to;\n\
      if (sel.shift || posEq(sel.from, sel.to)) pos = findPosH(this, dir, unit, true);\n\
      setSelectionUser(this, pos, pos);\n\
    }),\n\
\n\
    deleteH: operation(null, function(dir, unit) {\n\
      var sel = this.view.sel;\n\
      if (!posEq(sel.from, sel.to)) replaceRange(this, \"\", sel.from, sel.to);\n\
      else replaceRange(this, \"\", sel.from, findPosH(this, dir, unit, false));\n\
      this.curOp.userSelChange = true;\n\
    }),\n\
\n\
    moveV: operation(null, function(dir, unit) {\n\
      var view = this.view, doc = view.doc, display = this.display;\n\
      var dist = 0, cur = selHead(view), pos = cursorCoords(this, cur, \"div\");\n\
      var x = pos.left, y;\n\
      if (view.goalColumn != null) x = view.goalColumn;\n\
      if (unit == \"page\") {\n\
        var pageSize = Math.min(display.wrapper.clientHeight, window.innerHeight || document.documentElement.clientHeight);\n\
        y = pos.top + dir * pageSize;\n\
      } else if (unit == \"line\") {\n\
        y = dir > 0 ? pos.bottom + 3 : pos.top - 3;\n\
      }\n\
      var target = coordsChar(this, x, y), line;\n\
      // Work around problem with moving 'through' line widgets\n\
      if (dir > 0 && target.line == cur.line && cur.line < doc.size - 1 && getLine(doc, cur.line).widgets &&\n\
          Math.abs(cursorCoords(this, target, \"div\").top - pos.top) < 2)\n\
        target = coordsChar(this, x, cursorCoords(this, {line: cur.line + 1, ch: 0}, \"div\").top + 3);\n\
      else if (dir < 0 && cur.line > 0 && (line = getLine(doc, target.line)).widgets && target.ch == line.text.length)\n\
        target = coordsChar(this, x, cursorCoords(this, {line: target.line, ch: line.text.length}, \"div\").bottom - 3);\n\
          \n\
      if (unit == \"page\") display.scrollbarV.scrollTop += charCoords(this, target, \"div\").top - pos.top;\n\
      setSelectionUser(this, target, target);\n\
      view.goalColumn = x;\n\
    }),\n\
\n\
    toggleOverwrite: function() {\n\
      if (this.view.overwrite = !this.view.overwrite)\n\
        this.display.cursor.className += \" CodeMirror-overwrite\";\n\
      else\n\
        this.display.cursor.className = this.display.cursor.className.replace(\" CodeMirror-overwrite\", \"\");\n\
    },\n\
\n\
    posFromIndex: function(off) {\n\
      var lineNo = 0, ch, doc = this.view.doc;\n\
      doc.iter(0, doc.size, function(line) {\n\
        var sz = line.text.length + 1;\n\
        if (sz > off) { ch = off; return true; }\n\
        off -= sz;\n\
        ++lineNo;\n\
      });\n\
      return clipPos(doc, {line: lineNo, ch: ch});\n\
    },\n\
    indexFromPos: function (coords) {\n\
      if (coords.line < 0 || coords.ch < 0) return 0;\n\
      var index = coords.ch;\n\
      this.view.doc.iter(0, coords.line, function (line) {\n\
        index += line.text.length + 1;\n\
      });\n\
      return index;\n\
    },\n\
\n\
    scrollTo: function(x, y) {\n\
      if (x != null) this.display.scrollbarH.scrollLeft = this.display.scroller.scrollLeft = x;\n\
      if (y != null) this.display.scrollbarV.scrollTop = this.display.scroller.scrollTop = y;\n\
      updateDisplay(this, []);\n\
    },\n\
    getScrollInfo: function() {\n\
      var scroller = this.display.scroller, co = scrollerCutOff;\n\
      return {left: scroller.scrollLeft, top: scroller.scrollTop,\n\
              height: scroller.scrollHeight - co, width: scroller.scrollWidth - co,\n\
              clientHeight: scroller.clientHeight - co, clientWidth: scroller.clientWidth - co};\n\
    },\n\
\n\
    setSize: function(width, height) {\n\
      function interpret(val) {\n\
        val = String(val);\n\
        return /^\\d+$/.test(val) ? val + \"px\" : val;\n\
      }\n\
      if (width != null) this.display.wrapper.style.width = interpret(width);\n\
      if (height != null) this.display.wrapper.style.height = interpret(height);\n\
      this.refresh();\n\
    },\n\
\n\
    on: function(type, f) {on(this, type, f);},\n\
    off: function(type, f) {off(this, type, f);},\n\
\n\
    operation: function(f){return operation(this, f)();},\n\
    compoundChange: function(f){return compoundChange(this, f);},\n\
\n\
    refresh: function(){\n\
      updateDisplay(this, true, this.view.scrollTop);\n\
      if (this.display.scrollbarV.scrollHeight > this.view.scrollTop)\n\
        this.display.scrollbarV.scrollTop = this.view.scrollTop;\n\
    },\n\
\n\
    getInputField: function(){return this.display.input;},\n\
    getWrapperElement: function(){return this.display.wrapper;},\n\
    getScrollerElement: function(){return this.display.scroller;},\n\
    getGutterElement: function(){return this.display.gutters;}\n\
  };\n\
\n\
  // OPTION DEFAULTS\n\
\n\
  // The default configuration options.\n\
  var defaults = CodeMirror.defaults = {\n\
    value: \"\",\n\
    mode: null,\n\
    theme: \"default\",\n\
    indentUnit: 2,\n\
    indentWithTabs: false,\n\
    smartIndent: true,\n\
    tabSize: 4,\n\
    keyMap: \"default\",\n\
    extraKeys: null,\n\
    electricChars: true,\n\
    autoClearEmptyLines: false,\n\
    onKeyEvent: null,\n\
    onDragEvent: null,\n\
    lineWrapping: false,\n\
    lineNumbers: false,\n\
    gutters: [],\n\
    fixedGutter: false,\n\
    firstLineNumber: 1,\n\
    readOnly: false,\n\
    dragDrop: true,\n\
    cursorBlinkRate: 530,\n\
    workTime: 100,\n\
    workDelay: 200,\n\
    pollInterval: 100,\n\
    undoDepth: 40,\n\
    tabindex: null,\n\
    autofocus: null,\n\
    lineNumberFormatter: function(integer) { return integer; }\n\
  };\n\
\n\
  // MODE DEFINITION AND QUERYING\n\
\n\
  // Known modes, by name and by MIME\n\
  var modes = CodeMirror.modes = {}, mimeModes = CodeMirror.mimeModes = {};\n\
\n\
  CodeMirror.defineMode = function(name, mode) {\n\
    if (!CodeMirror.defaults.mode && name != \"null\") CodeMirror.defaults.mode = name;\n\
    if (arguments.length > 2) {\n\
      mode.dependencies = [];\n\
      for (var i = 2; i < arguments.length; ++i) mode.dependencies.push(arguments[i]);\n\
    }\n\
    modes[name] = mode;\n\
  };\n\
\n\
  CodeMirror.defineMIME = function(mime, spec) {\n\
    mimeModes[mime] = spec;\n\
  };\n\
\n\
  CodeMirror.resolveMode = function(spec) {\n\
    if (typeof spec == \"string\" && mimeModes.hasOwnProperty(spec))\n\
      spec = mimeModes[spec];\n\
    else if (typeof spec == \"string\" && /^[\\w\\-]+\\/[\\w\\-]+\\+xml$/.test(spec))\n\
      return CodeMirror.resolveMode(\"application/xml\");\n\
    if (typeof spec == \"string\") return {name: spec};\n\
    else return spec || {name: \"null\"};\n\
  };\n\
\n\
  CodeMirror.getMode = function(options, spec) {\n\
    var spec = CodeMirror.resolveMode(spec);\n\
    var mfactory = modes[spec.name];\n\
    if (!mfactory) return CodeMirror.getMode(options, \"text/plain\");\n\
    var modeObj = mfactory(options, spec);\n\
    if (modeExtensions.hasOwnProperty(spec.name)) {\n\
      var exts = modeExtensions[spec.name];\n\
      for (var prop in exts) if (exts.hasOwnProperty(prop)) modeObj[prop] = exts[prop];\n\
    }\n\
    modeObj.name = spec.name;\n\
    return modeObj;\n\
  };\n\
\n\
  CodeMirror.defineMode(\"null\", function() {\n\
    return {token: function(stream) {stream.skipToEnd();}};\n\
  });\n\
  CodeMirror.defineMIME(\"text/plain\", \"null\");\n\
\n\
  var modeExtensions = CodeMirror.modeExtensions = {};\n\
  CodeMirror.extendMode = function(mode, properties) {\n\
    var exts = modeExtensions.hasOwnProperty(mode) ? modeExtensions[mode] : (modeExtensions[mode] = {});\n\
    for (var prop in properties) if (properties.hasOwnProperty(prop))\n\
      exts[prop] = properties[prop];\n\
  };\n\
\n\
  // EXTENSIONS\n\
\n\
  CodeMirror.defineExtension = function(name, func) {\n\
    CodeMirror.prototype[name] = func;\n\
  };\n\
  var optionHandlers = CodeMirror.optionHandlers = {};\n\
  CodeMirror.defineOption = function(name, deflt, handler) {\n\
    CodeMirror.defaults[name] = deflt;\n\
    optionHandlers[name] = handler;\n\
  };\n\
\n\
  // MODE STATE HANDLING\n\
\n\
  // Utility functions for working with state. Exported because modes\n\
  // sometimes need to do this.\n\
  function copyState(mode, state) {\n\
    if (state === true) return state;\n\
    if (mode.copyState) return mode.copyState(state);\n\
    var nstate = {};\n\
    for (var n in state) {\n\
      var val = state[n];\n\
      if (val instanceof Array) val = val.concat([]);\n\
      nstate[n] = val;\n\
    }\n\
    return nstate;\n\
  }\n\
  CodeMirror.copyState = copyState;\n\
\n\
  function startState(mode, a1, a2) {\n\
    return mode.startState ? mode.startState(a1, a2) : true;\n\
  }\n\
  CodeMirror.startState = startState;\n\
\n\
  CodeMirror.innerMode = function(mode, state) {\n\
    while (mode.innerMode) {\n\
      var info = mode.innerMode(state);\n\
      state = info.state;\n\
      mode = info.mode;\n\
    }\n\
    return info || {mode: mode, state: state};\n\
  };\n\
\n\
  // STANDARD COMMANDS\n\
\n\
  var commands = CodeMirror.commands = {\n\
    selectAll: function(cm) {cm.setSelection({line: 0, ch: 0}, {line: cm.lineCount() - 1});},\n\
    killLine: function(cm) {\n\
      var from = cm.getCursor(true), to = cm.getCursor(false), sel = !posEq(from, to);\n\
      if (!sel && cm.getLine(from.line).length == from.ch)\n\
        cm.replaceRange(\"\", from, {line: from.line + 1, ch: 0});\n\
      else cm.replaceRange(\"\", from, sel ? to : {line: from.line});\n\
    },\n\
    deleteLine: function(cm) {var l = cm.getCursor().line; cm.replaceRange(\"\", {line: l, ch: 0}, {line: l});},\n\
    undo: function(cm) {cm.undo();},\n\
    redo: function(cm) {cm.redo();},\n\
    goDocStart: function(cm) {cm.setCursor(0, 0, true);},\n\
    goDocEnd: function(cm) {cm.setSelection({line: cm.lineCount() - 1}, null, true);},\n\
    goLineStart: function(cm) {\n\
      var line = cm.getCursor().line;\n\
      cm.setCursor(line, lineStart(cm.getLineHandle(line)), true);\n\
    },\n\
    goLineStartSmart: function(cm) {\n\
      var cur = cm.getCursor(), line = cm.getLineHandle(cur.line), order = getOrder(line);\n\
      if (!order || order[0].level == 0) {\n\
        var firstNonWS = Math.max(0, line.text.search(/\\S/));\n\
        cm.setCursor(cur.line, cur.ch <= firstNonWS && cur.ch ? 0 : firstNonWS, true);\n\
      } else cm.setCursor(cur.line, lineStart(line), true);\n\
    },\n\
    goLineEnd: function(cm) {\n\
      var line = cm.getCursor().line;\n\
      cm.setCursor(line, lineEnd(cm.getLineHandle(line)), true);\n\
    },\n\
    goLineUp: function(cm) {cm.moveV(-1, \"line\");},\n\
    goLineDown: function(cm) {cm.moveV(1, \"line\");},\n\
    goPageUp: function(cm) {cm.moveV(-1, \"page\");},\n\
    goPageDown: function(cm) {cm.moveV(1, \"page\");},\n\
    goCharLeft: function(cm) {cm.moveH(-1, \"char\");},\n\
    goCharRight: function(cm) {cm.moveH(1, \"char\");},\n\
    goColumnLeft: function(cm) {cm.moveH(-1, \"column\");},\n\
    goColumnRight: function(cm) {cm.moveH(1, \"column\");},\n\
    goWordLeft: function(cm) {cm.moveH(-1, \"word\");},\n\
    goWordRight: function(cm) {cm.moveH(1, \"word\");},\n\
    delCharBefore: function(cm) {cm.deleteH(-1, \"char\");},\n\
    delCharAfter: function(cm) {cm.deleteH(1, \"char\");},\n\
    delWordBefore: function(cm) {cm.deleteH(-1, \"word\");},\n\
    delWordAfter: function(cm) {cm.deleteH(1, \"word\");},\n\
    indentAuto: function(cm) {cm.indentSelection(\"smart\");},\n\
    indentMore: function(cm) {cm.indentSelection(\"add\");},\n\
    indentLess: function(cm) {cm.indentSelection(\"subtract\");},\n\
    insertTab: function(cm) {cm.replaceSelection(\"\\t\", \"end\");},\n\
    defaultTab: function(cm) {\n\
      if (cm.somethingSelected()) cm.indentSelection(\"add\");\n\
      else cm.replaceSelection(\"\\t\", \"end\");\n\
    },\n\
    transposeChars: function(cm) {\n\
      var cur = cm.getCursor(), line = cm.getLine(cur.line);\n\
      if (cur.ch > 0 && cur.ch < line.length - 1)\n\
        cm.replaceRange(line.charAt(cur.ch) + line.charAt(cur.ch - 1),\n\
                        {line: cur.line, ch: cur.ch - 1}, {line: cur.line, ch: cur.ch + 1});\n\
    },\n\
    newlineAndIndent: function(cm) {\n\
      cm.replaceSelection(\"\\n\", \"end\");\n\
      cm.indentLine(cm.getCursor().line);\n\
    },\n\
    toggleOverwrite: function(cm) {cm.toggleOverwrite();}\n\
  };\n\
\n\
  // STANDARD KEYMAPS\n\
\n\
  var keyMap = CodeMirror.keyMap = {};\n\
  keyMap.basic = {\n\
    \"Left\": \"goCharLeft\", \"Right\": \"goCharRight\", \"Up\": \"goLineUp\", \"Down\": \"goLineDown\",\n\
    \"End\": \"goLineEnd\", \"Home\": \"goLineStartSmart\", \"PageUp\": \"goPageUp\", \"PageDown\": \"goPageDown\",\n\
    \"Delete\": \"delCharAfter\", \"Backspace\": \"delCharBefore\", \"Tab\": \"defaultTab\", \"Shift-Tab\": \"indentAuto\",\n\
    \"Enter\": \"newlineAndIndent\", \"Insert\": \"toggleOverwrite\"\n\
  };\n\
  // Note that the save and find-related commands aren't defined by\n\
  // default. Unknown commands are simply ignored.\n\
  keyMap.pcDefault = {\n\
    \"Ctrl-A\": \"selectAll\", \"Ctrl-D\": \"deleteLine\", \"Ctrl-Z\": \"undo\", \"Shift-Ctrl-Z\": \"redo\", \"Ctrl-Y\": \"redo\",\n\
    \"Ctrl-Home\": \"goDocStart\", \"Alt-Up\": \"goDocStart\", \"Ctrl-End\": \"goDocEnd\", \"Ctrl-Down\": \"goDocEnd\",\n\
    \"Ctrl-Left\": \"goWordLeft\", \"Ctrl-Right\": \"goWordRight\", \"Alt-Left\": \"goLineStart\", \"Alt-Right\": \"goLineEnd\",\n\
    \"Ctrl-Backspace\": \"delWordBefore\", \"Ctrl-Delete\": \"delWordAfter\", \"Ctrl-S\": \"save\", \"Ctrl-F\": \"find\",\n\
    \"Ctrl-G\": \"findNext\", \"Shift-Ctrl-G\": \"findPrev\", \"Shift-Ctrl-F\": \"replace\", \"Shift-Ctrl-R\": \"replaceAll\",\n\
    \"Ctrl-[\": \"indentLess\", \"Ctrl-]\": \"indentMore\",\n\
    fallthrough: \"basic\"\n\
  };\n\
  keyMap.macDefault = {\n\
    \"Cmd-A\": \"selectAll\", \"Cmd-D\": \"deleteLine\", \"Cmd-Z\": \"undo\", \"Shift-Cmd-Z\": \"redo\", \"Cmd-Y\": \"redo\",\n\
    \"Cmd-Up\": \"goDocStart\", \"Cmd-End\": \"goDocEnd\", \"Cmd-Down\": \"goDocEnd\", \"Alt-Left\": \"goWordLeft\",\n\
    \"Alt-Right\": \"goWordRight\", \"Cmd-Left\": \"goLineStart\", \"Cmd-Right\": \"goLineEnd\", \"Alt-Backspace\": \"delWordBefore\",\n\
    \"Ctrl-Alt-Backspace\": \"delWordAfter\", \"Alt-Delete\": \"delWordAfter\", \"Cmd-S\": \"save\", \"Cmd-F\": \"find\",\n\
    \"Cmd-G\": \"findNext\", \"Shift-Cmd-G\": \"findPrev\", \"Cmd-Alt-F\": \"replace\", \"Shift-Cmd-Alt-F\": \"replaceAll\",\n\
    \"Cmd-[\": \"indentLess\", \"Cmd-]\": \"indentMore\",\n\
    fallthrough: [\"basic\", \"emacsy\"]\n\
  };\n\
  keyMap[\"default\"] = mac ? keyMap.macDefault : keyMap.pcDefault;\n\
  keyMap.emacsy = {\n\
    \"Ctrl-F\": \"goCharRight\", \"Ctrl-B\": \"goCharLeft\", \"Ctrl-P\": \"goLineUp\", \"Ctrl-N\": \"goLineDown\",\n\
    \"Alt-F\": \"goWordRight\", \"Alt-B\": \"goWordLeft\", \"Ctrl-A\": \"goLineStart\", \"Ctrl-E\": \"goLineEnd\",\n\
    \"Ctrl-V\": \"goPageUp\", \"Shift-Ctrl-V\": \"goPageDown\", \"Ctrl-D\": \"delCharAfter\", \"Ctrl-H\": \"delCharBefore\",\n\
    \"Alt-D\": \"delWordAfter\", \"Alt-Backspace\": \"delWordBefore\", \"Ctrl-K\": \"killLine\", \"Ctrl-T\": \"transposeChars\"\n\
  };\n\
\n\
  // KEYMAP DISPATCH\n\
\n\
  function getKeyMap(val) {\n\
    if (typeof val == \"string\") return keyMap[val];\n\
    else return val;\n\
  }\n\
\n\
  function lookupKey(name, extraMap, map, handle, stop) {\n\
    function lookup(map) {\n\
      map = getKeyMap(map);\n\
      var found = map[name];\n\
      if (found === false) {\n\
        if (stop) stop();\n\
        return true;\n\
      }\n\
      if (found != null && handle(found)) return true;\n\
      if (map.nofallthrough) {\n\
        if (stop) stop();\n\
        return true;\n\
      }\n\
      var fallthrough = map.fallthrough;\n\
      if (fallthrough == null) return false;\n\
      if (Object.prototype.toString.call(fallthrough) != \"[object Array]\")\n\
        return lookup(fallthrough);\n\
      for (var i = 0, e = fallthrough.length; i < e; ++i) {\n\
        if (lookup(fallthrough[i])) return true;\n\
      }\n\
      return false;\n\
    }\n\
    if (extraMap && lookup(extraMap)) return true;\n\
    return lookup(map);\n\
  }\n\
  function isModifierKey(event) {\n\
    var name = keyNames[e_prop(event, \"keyCode\")];\n\
    return name == \"Ctrl\" || name == \"Alt\" || name == \"Shift\" || name == \"Mod\";\n\
  }\n\
\n\
  // FROMTEXTAREA\n\
\n\
  CodeMirror.fromTextArea = function(textarea, options) {\n\
    if (!options) options = {};\n\
    options.value = textarea.value;\n\
    if (!options.tabindex && textarea.tabindex)\n\
      options.tabindex = textarea.tabindex;\n\
    // Set autofocus to true if this textarea is focused, or if it has\n\
    // autofocus and no other element is focused.\n\
    if (options.autofocus == null) {\n\
      var hasFocus = document.body;\n\
      // doc.activeElement occasionally throws on IE\n\
      try { hasFocus = document.activeElement; } catch(e) {}\n\
      options.autofocus = hasFocus == textarea ||\n\
        textarea.getAttribute(\"autofocus\") != null && hasFocus == document.body;\n\
    }\n\
\n\
    function save() {textarea.value = cm.getValue();}\n\
    if (textarea.form) {\n\
      // Deplorable hack to make the submit method do the right thing.\n\
      on(textarea.form, \"submit\", save);\n\
      if (typeof textarea.form.submit == \"function\") {\n\
        var realSubmit = textarea.form.submit;\n\
        textarea.form.submit = function wrappedSubmit() {\n\
          save();\n\
          textarea.form.submit = realSubmit;\n\
          textarea.form.submit();\n\
          textarea.form.submit = wrappedSubmit;\n\
        };\n\
      }\n\
    }\n\
\n\
    textarea.style.display = \"none\";\n\
    var cm = CodeMirror(function(node) {\n\
      textarea.parentNode.insertBefore(node, textarea.nextSibling);\n\
    }, options);\n\
    cm.save = save;\n\
    cm.getTextArea = function() { return textarea; };\n\
    cm.toTextArea = function() {\n\
      save();\n\
      textarea.parentNode.removeChild(cm.getWrapperElement());\n\
      textarea.style.display = \"\";\n\
      if (textarea.form) {\n\
        off(textarea.form, \"submit\", save);\n\
        if (typeof textarea.form.submit == \"function\")\n\
          textarea.form.submit = realSubmit;\n\
      }\n\
    };\n\
    return cm;\n\
  };\n\
\n\
  // STRING STREAM\n\
\n\
  // Fed to the mode parsers, provides helper functions to make\n\
  // parsers more succinct.\n\
\n\
  // The character stream used by a mode's parser.\n\
  function StringStream(string, tabSize) {\n\
    this.pos = this.start = 0;\n\
    this.string = string;\n\
    this.tabSize = tabSize || 8;\n\
  }\n\
\n\
  StringStream.prototype = {\n\
    eol: function() {return this.pos >= this.string.length;},\n\
    sol: function() {return this.pos == 0;},\n\
    peek: function() {return this.string.charAt(this.pos) || undefined;},\n\
    next: function() {\n\
      if (this.pos < this.string.length)\n\
        return this.string.charAt(this.pos++);\n\
    },\n\
    eat: function(match) {\n\
      var ch = this.string.charAt(this.pos);\n\
      if (typeof match == \"string\") var ok = ch == match;\n\
      else var ok = ch && (match.test ? match.test(ch) : match(ch));\n\
      if (ok) {++this.pos; return ch;}\n\
    },\n\
    eatWhile: function(match) {\n\
      var start = this.pos;\n\
      while (this.eat(match)){}\n\
      return this.pos > start;\n\
    },\n\
    eatSpace: function() {\n\
      var start = this.pos;\n\
      while (/[\\s\\u00a0]/.test(this.string.charAt(this.pos))) ++this.pos;\n\
      return this.pos > start;\n\
    },\n\
    skipToEnd: function() {this.pos = this.string.length;},\n\
    skipTo: function(ch) {\n\
      var found = this.string.indexOf(ch, this.pos);\n\
      if (found > -1) {this.pos = found; return true;}\n\
    },\n\
    backUp: function(n) {this.pos -= n;},\n\
    column: function() {return countColumn(this.string, this.start, this.tabSize);},\n\
    indentation: function() {return countColumn(this.string, null, this.tabSize);},\n\
    match: function(pattern, consume, caseInsensitive) {\n\
      if (typeof pattern == \"string\") {\n\
        var cased = function(str) {return caseInsensitive ? str.toLowerCase() : str;};\n\
        if (cased(this.string).indexOf(cased(pattern), this.pos) == this.pos) {\n\
          if (consume !== false) this.pos += pattern.length;\n\
          return true;\n\
        }\n\
      } else {\n\
        var match = this.string.slice(this.pos).match(pattern);\n\
        if (match && match.index > 0) return null;\n\
        if (match && consume !== false) this.pos += match[0].length;\n\
        return match;\n\
      }\n\
    },\n\
    current: function(){return this.string.slice(this.start, this.pos);}\n\
  };\n\
  CodeMirror.StringStream = StringStream;\n\
\n\
  // TEXTMARKERS\n\
\n\
  function TextMarker(cm, type, style) {\n\
    this.lines = [];\n\
    this.type = type;\n\
    this.cm = cm;\n\
    if (style) this.style = style;\n\
  }\n\
\n\
  TextMarker.prototype.clear = function() {\n\
    startOperation(this.cm);\n\
    var min = Infinity, max = -Infinity;\n\
    for (var i = 0; i < this.lines.length; ++i) {\n\
      var line = this.lines[i];\n\
      var span = getMarkedSpanFor(line.markedSpans, this, true);\n\
      if (span.from != null || span.to != null) {\n\
        var lineN = lineNo(line);\n\
        min = Math.min(min, lineN); max = Math.max(max, lineN);\n\
      }\n\
    }\n\
    if (min != Infinity) regChange(this.cm, min, max + 1);\n\
    this.lines.length = 0;\n\
    endOperation(this.cm);\n\
  };\n\
\n\
  TextMarker.prototype.find = function() {\n\
    var from, to;\n\
    for (var i = 0; i < this.lines.length; ++i) {\n\
      var line = this.lines[i];\n\
      var span = getMarkedSpanFor(line.markedSpans, this);\n\
      if (span.from != null || span.to != null) {\n\
        var found = lineNo(line);\n\
        if (span.from != null) from = {line: found, ch: span.from};\n\
        if (span.to != null) to = {line: found, ch: span.to};\n\
      }\n\
    }\n\
    if (this.type == \"bookmark\") return from;\n\
    return from && {from: from, to: to};\n\
  };\n\
\n\
  // TEXTMARKER SPANS\n\
\n\
  function getMarkedSpanFor(spans, marker, del) {\n\
    if (spans) for (var i = 0; i < spans.length; ++i) {\n\
      var span = spans[i];\n\
      if (span.marker == marker) {\n\
        if (del) spans.splice(i, 1);\n\
        return span;\n\
      }\n\
    }\n\
  }\n\
\n\
  function markedSpansBefore(old, startCh, endCh) {\n\
    if (old) for (var i = 0, nw; i < old.length; ++i) {\n\
      var span = old[i], marker = span.marker;\n\
      var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= startCh : span.from < startCh);\n\
      if (startsBefore || marker.type == \"bookmark\" && span.from == startCh && span.from != endCh) {\n\
        var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= startCh : span.to > startCh);\n\
        (nw || (nw = [])).push({from: span.from,\n\
                                to: endsAfter ? null : span.to,\n\
                                marker: marker});\n\
      }\n\
    }\n\
    return nw;\n\
  }\n\
\n\
  function markedSpansAfter(old, endCh) {\n\
    if (old) for (var i = 0, nw; i < old.length; ++i) {\n\
      var span = old[i], marker = span.marker;\n\
      var endsAfter = span.to == null || (marker.inclusiveRight ? span.to >= endCh : span.to > endCh);\n\
      if (endsAfter || marker.type == \"bookmark\" && span.from == endCh) {\n\
        var startsBefore = span.from == null || (marker.inclusiveLeft ? span.from <= endCh : span.from < endCh);\n\
        (nw || (nw = [])).push({from: startsBefore ? null : span.from - endCh,\n\
                                to: span.to == null ? null : span.to - endCh,\n\
                                marker: marker});\n\
      }\n\
    }\n\
    return nw;\n\
  }\n\
\n\
  function updateMarkedSpans(oldFirst, oldLast, startCh, endCh, newText) {\n\
    if (!oldFirst && !oldLast) return newText;\n\
    // Get the spans that 'stick out' on both sides\n\
    var first = markedSpansBefore(oldFirst, startCh);\n\
    var last = markedSpansAfter(oldLast, endCh);\n\
\n\
    // Next, merge those two ends\n\
    var sameLine = newText.length == 1, offset = lst(newText).length + (sameLine ? startCh : 0);\n\
    if (first) {\n\
      // Fix up .to properties of first\n\
      for (var i = 0; i < first.length; ++i) {\n\
        var span = first[i];\n\
        if (span.to == null) {\n\
          var found = getMarkedSpanFor(last, span.marker);\n\
          if (!found) span.to = startCh;\n\
          else if (sameLine) span.to = found.to == null ? null : found.to + offset;\n\
        }\n\
      }\n\
    }\n\
    if (last) {\n\
      // Fix up .from in last (or move them into first in case of sameLine)\n\
      for (var i = 0; i < last.length; ++i) {\n\
        var span = last[i];\n\
        if (span.to != null) span.to += offset;\n\
        if (span.from == null) {\n\
          var found = getMarkedSpanFor(first, span.marker);\n\
          if (!found) {\n\
            span.from = offset;\n\
            if (sameLine) (first || (first = [])).push(span);\n\
          }\n\
        } else {\n\
          span.from += offset;\n\
          if (sameLine) (first || (first = [])).push(span);\n\
        }\n\
      }\n\
    }\n\
\n\
    var newMarkers = [newHL(newText[0], first)];\n\
    if (!sameLine) {\n\
      // Fill gap with whole-line-spans\n\
      var gap = newText.length - 2, gapMarkers;\n\
      if (gap > 0 && first)\n\
        for (var i = 0; i < first.length; ++i)\n\
          if (first[i].to == null)\n\
            (gapMarkers || (gapMarkers = [])).push({from: null, to: null, marker: first[i].marker});\n\
      for (var i = 0; i < gap; ++i)\n\
        newMarkers.push(newHL(newText[i+1], gapMarkers));\n\
      newMarkers.push(newHL(lst(newText), last));\n\
    }\n\
    return newMarkers;\n\
  }\n\
\n\
  // hl stands for history-line, a data structure that can be either a\n\
  // string (line without markers) or a {text, markedSpans} object.\n\
  function hlText(val) { return typeof val == \"string\" ? val : val.text; }\n\
  function hlSpans(val) { return typeof val == \"string\" ? null : val.markedSpans; }\n\
  function newHL(text, spans) { return spans ? {text: text, markedSpans: spans} : text; }\n\
\n\
  function detachMarkedSpans(line) {\n\
    var spans = line.markedSpans;\n\
    if (!spans) return;\n\
    for (var i = 0; i < spans.length; ++i) {\n\
      var lines = spans[i].marker.lines;\n\
      var ix = indexOf(lines, line);\n\
      lines.splice(ix, 1);\n\
    }\n\
    line.markedSpans = null;\n\
  }\n\
\n\
  function attachMarkedSpans(line, spans) {\n\
    if (!spans) return;\n\
    for (var i = 0; i < spans.length; ++i)\n\
      var marker = spans[i].marker.lines.push(line);\n\
    line.markedSpans = spans;\n\
  }\n\
\n\
  // LINE DATA STRUCTURE\n\
\n\
  // Line objects. These hold state related to a line, including\n\
  // highlighting info (the styles array).\n\
  function Line(text, markedSpans, height) {\n\
    this.text = text;\n\
    this.height = height;\n\
    attachMarkedSpans(this, markedSpans);\n\
  }\n\
\n\
  Line.prototype = {\n\
    update: function(text, markedSpans, cm) {\n\
      this.text = text;\n\
      this.stateAfter = this.styles = null;\n\
      detachMarkedSpans(this);\n\
      attachMarkedSpans(this, markedSpans);\n\
      signalLater(cm, this, \"change\");\n\
    },\n\
\n\
    // Run the given mode's parser over a line, update the styles\n\
    // array, which contains alternating fragments of text and CSS\n\
    // classes.\n\
    highlight: function(mode, state, tabSize) {\n\
      var stream = new StringStream(this.text, tabSize), st = this.styles || (this.styles = []);\n\
      var pos = st.length = 0;\n\
      if (this.text == \"\" && mode.blankLine) mode.blankLine(state);\n\
      while (!stream.eol()) {\n\
        var style = mode.token(stream, state), substr = stream.current();\n\
        stream.start = stream.pos;\n\
        if (pos && st[pos-1] == style) {\n\
          st[pos-2] += substr;\n\
        } else if (substr) {\n\
          st[pos++] = substr; st[pos++] = style;\n\
        }\n\
        // Give up when line is ridiculously long\n\
        if (stream.pos > 5000) {\n\
          st[pos++] = this.text.slice(stream.pos); st[pos++] = null;\n\
          break;\n\
        }\n\
      }\n\
    },\n\
\n\
    // Lightweight form of highlight -- proceed over this line and\n\
    // update state, but don't save a style array.\n\
    process: function(mode, state, tabSize) {\n\
      var stream = new StringStream(this.text, tabSize);\n\
      if (this.text == \"\" && mode.blankLine) mode.blankLine(state);\n\
      while (!stream.eol() && stream.pos <= 5000) {\n\
        mode.token(stream, state);\n\
        stream.start = stream.pos;\n\
      }\n\
    },\n\
\n\
    // Fetch the parser token for a given character. Useful for hacks\n\
    // that want to inspect the mode state (say, for completion).\n\
    getTokenAt: function(mode, state, tabSize, ch) {\n\
      var txt = this.text, stream = new StringStream(txt, tabSize);\n\
      while (stream.pos < ch && !stream.eol()) {\n\
        stream.start = stream.pos;\n\
        var style = mode.token(stream, state);\n\
      }\n\
      return {start: stream.start,\n\
              end: stream.pos,\n\
              string: stream.current(),\n\
              className: style || null,\n\
              state: state};\n\
    },\n\
\n\
    indentation: function(tabSize) {return countColumn(this.text, null, tabSize);},\n\
\n\
    // Produces an HTML fragment for the line, taking selection,\n\
    // marking, and highlighting into account.\n\
    getContent: function(tabSize, wrapAt, compensateForWrapping) {\n\
      var first = true, col = 0, specials = /[\\t\\u0000-\\u0019\\u200b\\u2028\\u2029\\uFEFF]/g;\n\
      var pre = elt(\"pre\");\n\
      function span_(text, style) {\n\
        if (!text) return;\n\
        // Work around a bug where, in some compat modes, IE ignores leading spaces\n\
        if (first && ie && text.charAt(0) == \" \") text = \"\\u00a0\" + text.slice(1);\n\
        first = false;\n\
        if (!specials.test(text)) {\n\
          col += text.length;\n\
          var content = document.createTextNode(text);\n\
        } else {\n\
          var content = document.createDocumentFragment(), pos = 0;\n\
          while (true) {\n\
            specials.lastIndex = pos;\n\
            var m = specials.exec(text);\n\
            var skipped = m ? m.index - pos : text.length - pos;\n\
            if (skipped) {\n\
              content.appendChild(document.createTextNode(text.slice(pos, pos + skipped)));\n\
              col += skipped;\n\
            }\n\
            if (!m) break;\n\
            pos += skipped + 1;\n\
            if (m[0] == \"\\t\") {\n\
              var tabWidth = tabSize - col % tabSize;\n\
              content.appendChild(elt(\"span\", spaceStr(tabWidth), \"cm-tab\"));\n\
              col += tabWidth;\n\
            } else {\n\
              var token = elt(\"span\", \"\\u2022\", \"cm-invalidchar\");\n\
              token.title = \"\\\\u\" + m[0].charCodeAt(0).toString(16);\n\
              content.appendChild(token);\n\
              col += 1;\n\
            }\n\
          }\n\
        }\n\
        if (style != null) return pre.appendChild(elt(\"span\", [content], style));\n\
        else return pre.appendChild(content);\n\
      }\n\
      var span = span_;\n\
      if (wrapAt != null) {\n\
        var outPos = 0;\n\
        span = function(text, style) {\n\
          var l = text.length;\n\
          if (wrapAt >= outPos && wrapAt < outPos + l) {\n\
            var cut = wrapAt - outPos;\n\
            if (wrapAt > outPos) {\n\
              span_(text.slice(0, cut), style);\n\
              // See comment at the definition of spanAffectsWrapping\n\
              if (compensateForWrapping && spanAffectsWrapping.test(text.slice(cut - 1, cut + 1)))\n\
                pre.appendChild(elt(\"wbr\"));\n\
            }\n\
            if (cut + 1 == l) {\n\
              pre.anchor = span_(text.slice(cut), style || \"\");\n\
              wrapAt--;\n\
            } else {\n\
              var end = cut + 1;\n\
              while (isExtendingChar.test(text.charAt(end))) ++end;\n\
              pre.anchor = span_(text.slice(cut, end), style || \"\");\n\
              if (compensateForWrapping && spanAffectsWrapping.test(text.slice(cut, end + 1)))\n\
                pre.appendChild(elt(\"wbr\"));\n\
              span_(text.slice(end), style);\n\
            }\n\
            outPos += l;\n\
          } else {\n\
            outPos += l;\n\
            span_(text, style);\n\
          }\n\
        };\n\
      }\n\
\n\
      var st = this.styles, allText = this.text, marked = this.markedSpans;\n\
      var len = allText.length;\n\
      function styleToClass(style) {\n\
        if (!style) return null;\n\
        return \"cm-\" + style.replace(/ +/g, \" cm-\");\n\
      }\n\
      if (!allText) {\n\
        span(\"\\u00a0\");\n\
      } else if (!marked || !marked.length) {\n\
        for (var i = 0, ch = 0; ch < len; i+=2) {\n\
          var str = st[i], style = st[i+1], l = str.length;\n\
          if (ch + l > len) str = str.slice(0, len - ch);\n\
          ch += l;\n\
          span(str, styleToClass(style));\n\
        }\n\
      } else {\n\
        marked.sort(function(a, b) { return a.from - b.from; });\n\
        var pos = 0, i = 0, text = \"\", style, sg = 0;\n\
        var nextChange = marked[0].from || 0, marks = [], markpos = 0;\n\
        var advanceMarks = function() {\n\
          var m;\n\
          while (markpos < marked.length &&\n\
                 ((m = marked[markpos]).from == pos || m.from == null)) {\n\
            if (m.marker.type == \"range\") marks.push(m);\n\
            ++markpos;\n\
          }\n\
          nextChange = markpos < marked.length ? marked[markpos].from : Infinity;\n\
          for (var i = 0; i < marks.length; ++i) {\n\
            var to = marks[i].to;\n\
            if (to == null) to = Infinity;\n\
            if (to == pos) marks.splice(i--, 1);\n\
            else nextChange = Math.min(to, nextChange);\n\
          }\n\
        };\n\
        var m = 0;\n\
        while (pos < len) {\n\
          if (nextChange == pos) advanceMarks();\n\
          var upto = Math.min(len, nextChange);\n\
          while (true) {\n\
            if (text) {\n\
              var end = pos + text.length;\n\
              var appliedStyle = style;\n\
              for (var j = 0; j < marks.length; ++j) {\n\
                var mark = marks[j];\n\
                appliedStyle = (appliedStyle ? appliedStyle + \" \" : \"\") + mark.marker.style;\n\
                if (mark.marker.endStyle && mark.to === Math.min(end, upto)) appliedStyle += \" \" + mark.marker.endStyle;\n\
                if (mark.marker.startStyle && mark.from === pos) appliedStyle += \" \" + mark.marker.startStyle;\n\
              }\n\
              span(end > upto ? text.slice(0, upto - pos) : text, appliedStyle);\n\
              if (end >= upto) {text = text.slice(upto - pos); pos = upto; break;}\n\
              pos = end;\n\
            }\n\
            text = st[i++]; style = styleToClass(st[i++]);\n\
          }\n\
        }\n\
      }\n\
      return pre;\n\
    },\n\
\n\
    cleanUp: function() {\n\
      this.parent = null;\n\
      detachMarkedSpans(this);\n\
    }\n\
  };\n\
\n\
  // DOCUMENT DATA STRUCTURE\n\
\n\
  function LeafChunk(lines) {\n\
    this.lines = lines;\n\
    this.parent = null;\n\
    for (var i = 0, e = lines.length, height = 0; i < e; ++i) {\n\
      lines[i].parent = this;\n\
      height += lines[i].height;\n\
    }\n\
    this.height = height;\n\
  }\n\
\n\
  LeafChunk.prototype = {\n\
    chunkSize: function() { return this.lines.length; },\n\
    remove: function(at, n, cm) {\n\
      for (var i = at, e = at + n; i < e; ++i) {\n\
        var line = this.lines[i];\n\
        this.height -= line.height;\n\
        line.cleanUp();\n\
        signalLater(cm, line, \"delete\");\n\
      }\n\
      this.lines.splice(at, n);\n\
    },\n\
    collapse: function(lines) {\n\
      lines.splice.apply(lines, [lines.length, 0].concat(this.lines));\n\
    },\n\
    insertHeight: function(at, lines, height) {\n\
      this.height += height;\n\
      this.lines = this.lines.slice(0, at).concat(lines).concat(this.lines.slice(at));\n\
      for (var i = 0, e = lines.length; i < e; ++i) lines[i].parent = this;\n\
    },\n\
    iterN: function(at, n, op) {\n\
      for (var e = at + n; at < e; ++at)\n\
        if (op(this.lines[at])) return true;\n\
    }\n\
  };\n\
\n\
  function BranchChunk(children) {\n\
    this.children = children;\n\
    var size = 0, height = 0;\n\
    for (var i = 0, e = children.length; i < e; ++i) {\n\
      var ch = children[i];\n\
      size += ch.chunkSize(); height += ch.height;\n\
      ch.parent = this;\n\
    }\n\
    this.size = size;\n\
    this.height = height;\n\
    this.parent = null;\n\
  }\n\
\n\
  BranchChunk.prototype = {\n\
    chunkSize: function() { return this.size; },\n\
    remove: function(at, n, callbacks) {\n\
      this.size -= n;\n\
      for (var i = 0; i < this.children.length; ++i) {\n\
        var child = this.children[i], sz = child.chunkSize();\n\
        if (at < sz) {\n\
          var rm = Math.min(n, sz - at), oldHeight = child.height;\n\
          child.remove(at, rm, callbacks);\n\
          this.height -= oldHeight - child.height;\n\
          if (sz == rm) { this.children.splice(i--, 1); child.parent = null; }\n\
          if ((n -= rm) == 0) break;\n\
          at = 0;\n\
        } else at -= sz;\n\
      }\n\
      if (this.size - n < 25) {\n\
        var lines = [];\n\
        this.collapse(lines);\n\
        this.children = [new LeafChunk(lines)];\n\
        this.children[0].parent = this;\n\
      }\n\
    },\n\
    collapse: function(lines) {\n\
      for (var i = 0, e = this.children.length; i < e; ++i) this.children[i].collapse(lines);\n\
    },\n\
    insert: function(at, lines) {\n\
      var height = 0;\n\
      for (var i = 0, e = lines.length; i < e; ++i) height += lines[i].height;\n\
      this.insertHeight(at, lines, height);\n\
    },\n\
    insertHeight: function(at, lines, height) {\n\
      this.size += lines.length;\n\
      this.height += height;\n\
      for (var i = 0, e = this.children.length; i < e; ++i) {\n\
        var child = this.children[i], sz = child.chunkSize();\n\
        if (at <= sz) {\n\
          child.insertHeight(at, lines, height);\n\
          if (child.lines && child.lines.length > 50) {\n\
            while (child.lines.length > 50) {\n\
              var spilled = child.lines.splice(child.lines.length - 25, 25);\n\
              var newleaf = new LeafChunk(spilled);\n\
              child.height -= newleaf.height;\n\
              this.children.splice(i + 1, 0, newleaf);\n\
              newleaf.parent = this;\n\
            }\n\
            this.maybeSpill();\n\
          }\n\
          break;\n\
        }\n\
        at -= sz;\n\
      }\n\
    },\n\
    maybeSpill: function() {\n\
      if (this.children.length <= 10) return;\n\
      var me = this;\n\
      do {\n\
        var spilled = me.children.splice(me.children.length - 5, 5);\n\
        var sibling = new BranchChunk(spilled);\n\
        if (!me.parent) { // Become the parent node\n\
          var copy = new BranchChunk(me.children);\n\
          copy.parent = me;\n\
          me.children = [copy, sibling];\n\
          me = copy;\n\
        } else {\n\
          me.size -= sibling.size;\n\
          me.height -= sibling.height;\n\
          var myIndex = indexOf(me.parent.children, me);\n\
          me.parent.children.splice(myIndex + 1, 0, sibling);\n\
        }\n\
        sibling.parent = me.parent;\n\
      } while (me.children.length > 10);\n\
      me.parent.maybeSpill();\n\
    },\n\
    iter: function(from, to, op) { this.iterN(from, to - from, op); },\n\
    iterN: function(at, n, op) {\n\
      for (var i = 0, e = this.children.length; i < e; ++i) {\n\
        var child = this.children[i], sz = child.chunkSize();\n\
        if (at < sz) {\n\
          var used = Math.min(n, sz - at);\n\
          if (child.iterN(at, used, op)) return true;\n\
          if ((n -= used) == 0) break;\n\
          at = 0;\n\
        } else at -= sz;\n\
      }\n\
    }\n\
  };\n\
\n\
  // LINE UTILITIES\n\
\n\
  function lineDoc(line) {\n\
    for (var d = line.parent; d && d.parent; d = d.parent) {}\n\
    return d;\n\
  }\n\
\n\
  function getLine(chunk, n) {\n\
    while (!chunk.lines) {\n\
      for (var i = 0;; ++i) {\n\
        var child = chunk.children[i], sz = child.chunkSize();\n\
        if (n < sz) { chunk = child; break; }\n\
        n -= sz;\n\
      }\n\
    }\n\
    return chunk.lines[n];\n\
  }\n\
\n\
  function updateLineHeight(line, height) {\n\
    var diff = height - line.height;\n\
    for (var n = line; n; n = n.parent) n.height += diff;\n\
  }\n\
\n\
  function lineNo(line) {\n\
    if (line.parent == null) return null;\n\
    var cur = line.parent, no = indexOf(cur.lines, line);\n\
    for (var chunk = cur.parent; chunk; cur = chunk, chunk = chunk.parent) {\n\
      for (var i = 0, e = chunk.children.length; ; ++i) {\n\
        if (chunk.children[i] == cur) break;\n\
        no += chunk.children[i].chunkSize();\n\
      }\n\
    }\n\
    return no;\n\
  }\n\
\n\
  function lineAtHeight(chunk, h) {\n\
    var n = 0;\n\
    outer: do {\n\
      for (var i = 0, e = chunk.children.length; i < e; ++i) {\n\
        var child = chunk.children[i], ch = child.height;\n\
        if (h < ch) { chunk = child; continue outer; }\n\
        h -= ch;\n\
        n += child.chunkSize();\n\
      }\n\
      return n;\n\
    } while (!chunk.lines);\n\
    for (var i = 0, e = chunk.lines.length; i < e; ++i) {\n\
      var line = chunk.lines[i], lh = line.height;\n\
      if (h < lh) break;\n\
      h -= lh;\n\
    }\n\
    return n + i;\n\
  }\n\
\n\
  function heightAtLine(chunk, n) {\n\
    var h = 0;\n\
    outer: do {\n\
      for (var i = 0, e = chunk.children.length; i < e; ++i) {\n\
        var child = chunk.children[i], sz = child.chunkSize();\n\
        if (n < sz) { chunk = child; continue outer; }\n\
        n -= sz;\n\
        h += child.height;\n\
      }\n\
      return h;\n\
    } while (!chunk.lines);\n\
    for (var i = 0; i < n; ++i) h += chunk.lines[i].height;\n\
    return h;\n\
  }\n\
\n\
  function getOrder(line) {\n\
    var order = line.order;\n\
    if (order == null) order = line.order = bidiOrdering(line.text);\n\
    return order;\n\
  }\n\
\n\
  function lineContent(cm, line, anchorAt) {\n\
    if (!line.styles) {\n\
      var doc = lineDoc(line);\n\
      line.highlight(doc.mode, line.stateAfter = getStateBefore(doc, lineNo(line)), cm.options.tabSize);\n\
    }\n\
    return line.getContent(cm.options.tabSize, anchorAt, cm.options.lineWrapping);\n\
  }\n\
\n\
  // HISTORY\n\
\n\
  // The history object 'chunks' changes that are made close together\n\
  // and at almost the same time into bigger undoable units.\n\
  function History() {\n\
    this.time = 0;\n\
    this.done = []; this.undone = [];\n\
    this.compound = 0;\n\
    this.closed = false;\n\
  }\n\
  History.prototype = {\n\
    addChange: function(start, added, old) {\n\
      this.undone.length = 0;\n\
      var time = +new Date, cur = lst(this.done), last = cur && lst(cur);\n\
      var dtime = time - this.time;\n\
\n\
      if (this.compound && cur && !this.closed) {\n\
        cur.push({start: start, added: added, old: old});\n\
      } else if (dtime > 400 || !last || this.closed ||\n\
                 last.start > start + old.length || last.start + last.added < start) {\n\
        this.done.push([{start: start, added: added, old: old}]);\n\
        this.closed = false;\n\
      } else {\n\
        var startBefore = Math.max(0, last.start - start),\n\
            endAfter = Math.max(0, (start + old.length) - (last.start + last.added));\n\
        for (var i = startBefore; i > 0; --i) last.old.unshift(old[i - 1]);\n\
        for (var i = endAfter; i > 0; --i) last.old.push(old[old.length - i]);\n\
        if (startBefore) last.start = start;\n\
        last.added += added - (old.length - startBefore - endAfter);\n\
      }\n\
      this.time = time;\n\
    },\n\
    startCompound: function() {\n\
      if (!this.compound++) this.closed = true;\n\
    },\n\
    endCompound: function() {\n\
      if (!--this.compound) this.closed = true;\n\
    }\n\
  };\n\
\n\
  // EVENT OPERATORS\n\
\n\
  function stopMethod() {e_stop(this);}\n\
  // Ensure an event has a stop method.\n\
  function addStop(event) {\n\
    if (!event.stop) event.stop = stopMethod;\n\
    return event;\n\
  }\n\
\n\
  function e_preventDefault(e) {\n\
    if (e.preventDefault) e.preventDefault();\n\
    else e.returnValue = false;\n\
  }\n\
  function e_stopPropagation(e) {\n\
    if (e.stopPropagation) e.stopPropagation();\n\
    else e.cancelBubble = true;\n\
  }\n\
  function e_stop(e) {e_preventDefault(e); e_stopPropagation(e);}\n\
  CodeMirror.e_stop = e_stop;\n\
  CodeMirror.e_preventDefault = e_preventDefault;\n\
  CodeMirror.e_stopPropagation = e_stopPropagation;\n\
\n\
  function e_target(e) {return e.target || e.srcElement;}\n\
  function e_button(e) {\n\
    var b = e.which;\n\
    if (b == null) {\n\
      if (e.button & 1) b = 1;\n\
      else if (e.button & 2) b = 3;\n\
      else if (e.button & 4) b = 2;\n\
    }\n\
    if (mac && e.ctrlKey && b == 1) b = 3;\n\
    return b;\n\
  }\n\
\n\
  // Allow 3rd-party code to override event properties by adding an override\n\
  // object to an event object.\n\
  function e_prop(e, prop) {\n\
    var overridden = e.override && e.override.hasOwnProperty(prop);\n\
    return overridden ? e.override[prop] : e[prop];\n\
  }\n\
\n\
  // EVENT HANDLING\n\
\n\
  function on(emitter, type, f) {\n\
    if (emitter.addEventListener)\n\
      emitter.addEventListener(type, f, false);\n\
    else if (emitter.attachEvent)\n\
      emitter.attachEvent(\"on\" + type, f);\n\
    else {\n\
      var map = emitter._handlers || (emitter._handlers = {});\n\
      var arr = map[type] || (map[type] = []);\n\
      arr.push(f);\n\
    }\n\
  }\n\
\n\
  function off(emitter, type, f) {\n\
    if (emitter.removeEventListener)\n\
      emitter.removeEventListener(type, f, false);\n\
    else if (emitter.detachEvent)\n\
      emitter.detachEvent(\"on\" + type, f);\n\
    else {\n\
      var arr = emitter._handlers && emitter._handlers[type];\n\
      if (!arr) return;\n\
      for (var i = 0; i < arr.length; ++i)\n\
        if (arr[i] == f) { arr.splice(i, 1); break; }\n\
    }\n\
  }\n\
\n\
  function signal(emitter, type /*, values...*/) {\n\
    var arr = emitter._handlers && emitter._handlers[type];\n\
    if (!arr) return;\n\
    var args = Array.prototype.slice.call(arguments, 2);\n\
    for (var i = 0; i < arr.length; ++i) arr[i].apply(null, args);\n\
  }\n\
\n\
  function signalLater(cm, emitter, type /*, values...*/) {\n\
    var arr = emitter._handlers && emitter._handlers[type];\n\
    if (!arr) return;\n\
    var args = Array.prototype.slice.call(arguments, 3), flist = cm.curOp && cm.curOp.delayedCallbacks;\n\
    function bnd(f) {return function(){f.apply(null, args);};};\n\
    for (var i = 0; i < arr.length; ++i)\n\
      if (flist) flist.push(bnd(arr[i]));\n\
      else arr[i].apply(null, args);\n\
  }\n\
\n\
  function hasHandler(emitter, type) {\n\
    var arr = emitter._handlers && emitter._handlers[type];\n\
    return arr && arr.length > 0;\n\
  }\n\
\n\
  CodeMirror.on = on; CodeMirror.off = off; CodeMirror.signal = signal;\n\
\n\
  // MISC UTILITIES\n\
\n\
  // Number of pixels added to scroller and sizer to hide scrollbar\n\
  var scrollerCutOff = 30;\n\
\n\
  // Returned or thrown by various protocols to signal 'I'm not\n\
  // handling this'.\n\
  var Pass = CodeMirror.Pass = {toString: function(){return \"CodeMirror.Pass\";}};\n\
\n\
  function Delayed() {this.id = null;}\n\
  Delayed.prototype = {set: function(ms, f) {clearTimeout(this.id); this.id = setTimeout(f, ms);}};\n\
\n\
  // Counts the column offset in a string, taking tabs into account.\n\
  // Used mostly to find indentation.\n\
  function countColumn(string, end, tabSize) {\n\
    if (end == null) {\n\
      end = string.search(/[^\\s\\u00a0]/);\n\
      if (end == -1) end = string.length;\n\
    }\n\
    for (var i = 0, n = 0; i < end; ++i) {\n\
      if (string.charAt(i) == \"\\t\") n += tabSize - (n % tabSize);\n\
      else ++n;\n\
    }\n\
    return n;\n\
  }\n\
\n\
  var spaceStrs = [\"\"];\n\
  function spaceStr(n) {\n\
    while (spaceStrs.length <= n)\n\
      spaceStrs.push(lst(spaceStrs) + \" \");\n\
    return spaceStrs[n];\n\
  }\n\
\n\
  function lst(arr) { return arr[arr.length-1]; }\n\
\n\
  function selectInput(node) {\n\
    if (ios) { // Mobile Safari apparently has a bug where select() is broken.\n\
      node.selectionStart = 0;\n\
      node.selectionEnd = node.value.length;\n\
    } else node.select();\n\
  }\n\
\n\
  // Used to position the cursor after an undo/redo by finding the\n\
  // last edited character.\n\
  function editEnd(from, to) {\n\
    if (!to) return 0;\n\
    if (!from) return to.length;\n\
    for (var i = from.length, j = to.length; i >= 0 && j >= 0; --i, --j)\n\
      if (from.charAt(i) != to.charAt(j)) break;\n\
    return j + 1;\n\
  }\n\
\n\
  function indexOf(collection, elt) {\n\
    if (collection.indexOf) return collection.indexOf(elt);\n\
    for (var i = 0, e = collection.length; i < e; ++i)\n\
      if (collection[i] == elt) return i;\n\
    return -1;\n\
  }\n\
\n\
  function bind(f) {\n\
    var args = Array.prototype.slice.call(arguments, 1);\n\
    return function(){return f.apply(null, args);};\n\
  }\n\
\n\
  function isWordChar(ch) {\n\
    return /\\w/.test(ch) || ch.toUpperCase() != ch.toLowerCase();\n\
  }\n\
\n\
  function isEmpty(obj) {\n\
    var c = 0;\n\
    for (var n in obj) if (obj.hasOwnProperty(n) && obj[n]) ++c;\n\
    return !c;\n\
  }\n\
\n\
  var isExtendingChar = /[\\u0300-\\u036F\\u0483-\\u0487\\u0488-\\u0489\\u0591-\\u05BD\\u05BF\\u05C1-\\u05C2\\u05C4-\\u05C5\\u05C7\\u0610-\\u061A\\u064B-\\u065F\\u0670\\u06D6-\\u06DC\\u06DF-\\u06E4\\u06E7-\\u06E8\\u06EA-\\u06ED\\uA66F\\uA670-\\uA672\\uA674-\\uA67D\\uA69F]/;\n\
\n\
  // DOM UTILITIES\n\
\n\
  function elt(tag, content, className, style) {\n\
    var e = document.createElement(tag);\n\
    if (className) e.className = className;\n\
    if (style) e.style.cssText = style;\n\
    if (typeof content == \"string\") setTextContent(e, content);\n\
    else if (content) for (var i = 0; i < content.length; ++i) e.appendChild(content[i]);\n\
    return e;\n\
  }\n\
\n\
  function removeChildren(e) {\n\
    e.innerHTML = \"\";\n\
    return e;\n\
  }\n\
\n\
  function removeChildrenAndAdd(parent, e) {\n\
    return removeChildren(parent).appendChild(e);\n\
  }\n\
\n\
  function setTextContent(e, str) {\n\
    if (ie_lt9) {\n\
      e.innerHTML = \"\";\n\
      e.appendChild(document.createTextNode(str));\n\
    } else e.textContent = str;\n\
  }\n\
\n\
  // FEATURE DETECTION\n\
\n\
  // Detect drag-and-drop\n\
  var dragAndDrop = function() {\n\
    // There is *some* kind of drag-and-drop support in IE6-8, but I\n\
    // couldn't get it to work yet.\n\
    if (ie_lt9) return false;\n\
    var div = elt('div');\n\
    return \"draggable\" in div || \"dragDrop\" in div;\n\
  }();\n\
\n\
  // Feature-detect whether newlines in textareas are converted to \\r\\n\n\
  var lineSep = function () {\n\
    var te = elt(\"textarea\");\n\
    te.value = \"foo\\nbar\";\n\
    if (te.value.indexOf(\"\\r\") > -1) return \"\\r\\n\";\n\
    return \"\\n\";\n\
  }();\n\
\n\
  // For a reason I have yet to figure out, some browsers disallow\n\
  // word wrapping between certain characters *only* if a new inline\n\
  // element is started between them. This makes it hard to reliably\n\
  // measure the position of things, since that requires inserting an\n\
  // extra span. This terribly fragile set of regexps matches the\n\
  // character combinations that suffer from this phenomenon on the\n\
  // various browsers.\n\
  var spanAffectsWrapping = /^$/; // Won't match any two-character string\n\
  if (gecko) spanAffectsWrapping = /$'/;\n\
  else if (safari) spanAffectsWrapping = /\\-[^ \\-?]|\\?[^ !'\\\"\\),.\\-\\/:;\\?\\]\\}]/;\n\
  else if (chrome) spanAffectsWrapping = /\\-[^ \\-\\.?]|\\?[^ \\-\\.?\\]\\}:;!'\\\"\\),\\/]|[\\.!\\\"#&%\\)*+,:;=>\\]|\\}~][\\(\\{\\[<]|\\$'/;\n\
\n\
  var knownScrollbarWidth;\n\
  function scrollbarWidth(measure) {\n\
    if (knownScrollbarWidth != null) return knownScrollbarWidth;\n\
    var test = elt(\"div\", null, null, \"width: 50px; height: 50px; overflow-x: scroll\");\n\
    removeChildrenAndAdd(measure, test);\n\
    if (test.offsetWidth)\n\
      knownScrollbarWidth = test.offsetHeight - test.clientHeight;\n\
    return knownScrollbarWidth || 0;\n\
  }\n\
\n\
  // See if \"\".split is the broken IE version, if so, provide an\n\
  // alternative way to split lines.\n\
  var splitLines = \"\\n\\nb\".split(/\\n/).length != 3 ? function(string) {\n\
    var pos = 0, result = [], l = string.length;\n\
    while (pos <= l) {\n\
      var nl = string.indexOf(\"\\n\", pos);\n\
      if (nl == -1) nl = string.length;\n\
      var line = string.slice(pos, string.charAt(nl - 1) == \"\\r\" ? nl - 1 : nl);\n\
      var rt = line.indexOf(\"\\r\");\n\
      if (rt != -1) {\n\
        result.push(line.slice(0, rt));\n\
        pos += rt + 1;\n\
      } else {\n\
        result.push(line);\n\
        pos = nl + 1;\n\
      }\n\
    }\n\
    return result;\n\
  } : function(string){return string.split(/\\r\\n?|\\n/);};\n\
  CodeMirror.splitLines = splitLines;\n\
\n\
  var hasSelection = window.getSelection ? function(te) {\n\
    try { return te.selectionStart != te.selectionEnd; }\n\
    catch(e) { return false; }\n\
  } : function(te) {\n\
    try {var range = te.ownerDocument.selection.createRange();}\n\
    catch(e) {}\n\
    if (!range || range.parentElement() != te) return false;\n\
    return range.compareEndPoints(\"StartToEnd\", range) != 0;\n\
  };\n\
\n\
  var hasCopyEvent = (function() {\n\
    var e = elt(\"div\");\n\
    if (\"oncopy\" in e) return true;\n\
    e.setAttribute(\"oncopy\", \"return;\");\n\
    return typeof e.oncopy == 'function';\n\
  })();\n\
\n\
  // KEY NAMING\n\
\n\
  var keyNames = {3: \"Enter\", 8: \"Backspace\", 9: \"Tab\", 13: \"Enter\", 16: \"Shift\", 17: \"Ctrl\", 18: \"Alt\",\n\
                  19: \"Pause\", 20: \"CapsLock\", 27: \"Esc\", 32: \"Space\", 33: \"PageUp\", 34: \"PageDown\", 35: \"End\",\n\
                  36: \"Home\", 37: \"Left\", 38: \"Up\", 39: \"Right\", 40: \"Down\", 44: \"PrintScrn\", 45: \"Insert\",\n\
                  46: \"Delete\", 59: \";\", 91: \"Mod\", 92: \"Mod\", 93: \"Mod\", 109: \"-\", 107: \"=\", 127: \"Delete\",\n\
                  186: \";\", 187: \"=\", 188: \",\", 189: \"-\", 190: \".\", 191: \"/\", 192: \"`\", 219: \"[\", 220: \"\\\\\",\n\
                  221: \"]\", 222: \"'\", 63276: \"PageUp\", 63277: \"PageDown\", 63275: \"End\", 63273: \"Home\",\n\
                  63234: \"Left\", 63232: \"Up\", 63235: \"Right\", 63233: \"Down\", 63302: \"Insert\", 63272: \"Delete\"};\n\
  CodeMirror.keyNames = keyNames;\n\
  (function() {\n\
    // Number keys\n\
    for (var i = 0; i < 10; i++) keyNames[i + 48] = String(i);\n\
    // Alphabetic keys\n\
    for (var i = 65; i <= 90; i++) keyNames[i] = String.fromCharCode(i);\n\
    // Function keys\n\
    for (var i = 1; i <= 12; i++) keyNames[i + 111] = keyNames[i + 63235] = \"F\" + i;\n\
  })();\n\
\n\
  // BIDI HELPERS\n\
\n\
  function iterateBidiSections(order, from, to, f) {\n\
    if (!order) return f(from, to, \"ltr\");\n\
    for (var i = 0; i < order.length; ++i) {\n\
      var part = order[i];\n\
      if (part.from < to && part.to > from)\n\
        f(Math.max(part.from, from), Math.min(part.to, to), part.level == 1 ? \"rtl\" : \"ltr\");\n\
    }\n\
  }\n\
\n\
  function bidiLeft(part) { return part.level % 2 ? part.to : part.from; }\n\
  function bidiRight(part) { return part.level % 2 ? part.from : part.to; }\n\
\n\
  function lineLeft(line) { var order = getOrder(line); return order ? bidiLeft(order[0]) : 0; }\n\
  function lineRight(line) {\n\
    var order = getOrder(line);\n\
    if (!order) return line.text.length;\n\
    return bidiRight(lst(order));\n\
  }\n\
  function lineStart(line) {\n\
    var order = getOrder(line);\n\
    if (!order) return 0;\n\
    return order[0].level % 2 ? lineRight(line) : lineLeft(line);\n\
  }\n\
  function lineEnd(line) {\n\
    var order = getOrder(line);\n\
    if (!order) return line.text.length;\n\
    return order[0].level % 2 ? lineLeft(line) : lineRight(line);\n\
  }\n\
\n\
  // This is somewhat involved. It is needed in order to move\n\
  // 'visually' through bi-directional text -- i.e., pressing left\n\
  // should make the cursor go left, even when in RTL text. The\n\
  // tricky part is the 'jumps', where RTL and LTR text touch each\n\
  // other. This often requires the cursor offset to move more than\n\
  // one unit, in order to visually move one unit.\n\
  function moveVisually(line, start, dir, byUnit) {\n\
    var bidi = getOrder(line);\n\
    if (!bidi) return moveLogically(line, start, dir, byUnit);\n\
    var moveOneUnit = byUnit ? function(pos, dir) {\n\
      do pos += dir;\n\
      while (pos > 0 && isExtendingChar.test(line.text.charAt(pos)));\n\
      return pos;\n\
    } : function(pos, dir) { return pos + dir; };\n\
    var linedir = bidi[0].level;\n\
    for (var i = 0; i < bidi.length; ++i) {\n\
      var part = bidi[i], sticky = part.level % 2 == linedir;\n\
      if ((part.from < start && part.to > start) ||\n\
          (sticky && (part.from == start || part.to == start))) break;\n\
    }\n\
    var target = moveOneUnit(start, part.level % 2 ? -dir : dir);\n\
\n\
    while (target != null) {\n\
      if (part.level % 2 == linedir) {\n\
        if (target < part.from || target > part.to) {\n\
          part = bidi[i += dir];\n\
          target = part && (dir > 0 == part.level % 2 ? moveOneUnit(part.to, -1) : moveOneUnit(part.from, 1));\n\
        } else break;\n\
      } else {\n\
        if (target == bidiLeft(part)) {\n\
          part = bidi[--i];\n\
          target = part && bidiRight(part);\n\
        } else if (target == bidiRight(part)) {\n\
          part = bidi[++i];\n\
          target = part && bidiLeft(part);\n\
        } else break;\n\
      }\n\
    }\n\
\n\
    return target < 0 || target > line.text.length ? null : target;\n\
  }\n\
\n\
  function moveLogically(line, start, dir, byUnit) {\n\
    var target = start + dir;\n\
    if (byUnit) while (target > 0 && isExtendingChar.test(line.text.charAt(target))) target += dir;\n\
    return target < 0 || target > line.text.length ? null : target;\n\
  }\n\
\n\
  // Bidirectional ordering algorithm\n\
  // See http://unicode.org/reports/tr9/tr9-13.html for the algorithm\n\
  // that this (partially) implements.\n\
\n\
  // One-char codes used for character types:\n\
  // L (L):   Left-to-Right\n\
  // R (R):   Right-to-Left\n\
  // r (AL):  Right-to-Left Arabic\n\
  // 1 (EN):  European Number\n\
  // + (ES):  European Number Separator\n\
  // % (ET):  European Number Terminator\n\
  // n (AN):  Arabic Number\n\
  // , (CS):  Common Number Separator\n\
  // m (NSM): Non-Spacing Mark\n\
  // b (BN):  Boundary Neutral\n\
  // s (B):   Paragraph Separator\n\
  // t (S):   Segment Separator\n\
  // w (WS):  Whitespace\n\
  // N (ON):  Other Neutrals\n\
\n\
  // Returns null if characters are ordered as they appear\n\
  // (left-to-right), or an array of sections ({from, to, level}\n\
  // objects) in the order in which they occur visually.\n\
  var bidiOrdering = (function() {\n\
    // Character types for codepoints 0 to 0xff\n\
    var lowTypes = \"bbbbbbbbbtstwsbbbbbbbbbbbbbbssstwNN%%%NNNNNN,N,N1111111111NNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNbbbbbbsbbbbbbbbbbbbbbbbbbbbbbbbbb,N%%%%NNNNLNNNNN%%11NLNNN1LNNNNNLLLLLLLLLLLLLLLLLLLLLLLNLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLNLLLLLLLL\";\n\
    // Character types for codepoints 0x600 to 0x6ff\n\
    var arabicTypes = \"rrrrrrrrrrrr,rNNmmmmmmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmrrrrrrrnnnnnnnnnn%nnrrrmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmmmmmmNmmmmrrrrrrrrrrrrrrrrrr\";\n\
    function charType(code) {\n\
      var type = \"L\";\n\
      if (code <= 0xff) return lowTypes.charAt(code);\n\
      else if (0x590 <= code && code <= 0x5f4) return \"R\";\n\
      else if (0x600 <= code && code <= 0x6ff) return arabicTypes.charAt(code - 0x600);\n\
      else if (0x700 <= code && code <= 0x8ac) return \"r\";\n\
      else return \"L\";\n\
    }\n\
\n\
    var bidiRE = /[\\u0590-\\u05f4\\u0600-\\u06ff\\u0700-\\u08ac]/;\n\
    var isNeutral = /[stwN]/, isStrong = /[LRr]/, countsAsLeft = /[Lb1n]/, countsAsNum = /[1n]/;\n\
\n\
    return function charOrdering(str) {\n\
      if (!bidiRE.test(str)) return false;\n\
      var len = str.length, types = new Array(len), startType = null;\n\
      for (var i = 0; i < len; ++i) {\n\
        var type = types[i] = charType(str.charCodeAt(i));\n\
        if (startType == null) {\n\
          if (type == \"L\") startType = \"L\";\n\
          else if (type == \"R\" || type == \"r\") startType = \"R\";\n\
        }\n\
      }\n\
      if (startType == null) startType = \"L\";\n\
\n\
      // W1. Examine each non-spacing mark (NSM) in the level run, and\n\
      // change the type of the NSM to the type of the previous\n\
      // character. If the NSM is at the start of the level run, it will\n\
      // get the type of sor.\n\
      for (var i = 0, prev = startType; i < len; ++i) {\n\
        var type = types[i];\n\
        if (type == \"m\") types[i] = prev;\n\
        else prev = type;\n\
      }\n\
\n\
      // W2. Search backwards from each instance of a European number\n\
      // until the first strong type (R, L, AL, or sor) is found. If an\n\
      // AL is found, change the type of the European number to Arabic\n\
      // number.\n\
      // W3. Change all ALs to R.\n\
      for (var i = 0, cur = startType; i < len; ++i) {\n\
        var type = types[i];\n\
        if (type == \"1\" && cur == \"r\") types[i] = \"n\";\n\
        else if (isStrong.test(type)) { cur = type; if (type == \"r\") types[i] = \"R\"; }\n\
      }\n\
\n\
      // W4. A single European separator between two European numbers\n\
      // changes to a European number. A single common separator between\n\
      // two numbers of the same type changes to that type.\n\
      for (var i = 1, prev = types[0]; i < len - 1; ++i) {\n\
        var type = types[i];\n\
        if (type == \"+\" && prev == \"1\" && types[i+1] == \"1\") types[i] = \"1\";\n\
        else if (type == \",\" && prev == types[i+1] &&\n\
                 (prev == \"1\" || prev == \"n\")) types[i] = prev;\n\
        prev = type;\n\
      }\n\
\n\
      // W5. A sequence of European terminators adjacent to European\n\
      // numbers changes to all European numbers.\n\
      // W6. Otherwise, separators and terminators change to Other\n\
      // Neutral.\n\
      for (var i = 0; i < len; ++i) {\n\
        var type = types[i];\n\
        if (type == \",\") types[i] = \"N\";\n\
        else if (type == \"%\") {\n\
          for (var end = i + 1; end < len && types[end] == \"%\"; ++end) {}\n\
          var replace = (i && types[i-1] == \"!\") || (end < len - 1 && types[end] == \"1\") ? \"1\" : \"N\";\n\
          for (var j = i; j < end; ++j) types[j] = replace;\n\
          i = end - 1;\n\
        }\n\
      }\n\
\n\
      // W7. Search backwards from each instance of a European number\n\
      // until the first strong type (R, L, or sor) is found. If an L is\n\
      // found, then change the type of the European number to L.\n\
      for (var i = 0, cur = startType; i < len; ++i) {\n\
        var type = types[i];\n\
        if (cur == \"L\" && type == \"1\") types[i] = \"L\";\n\
        else if (isStrong.test(type)) cur = type;\n\
      }\n\
\n\
      // N1. A sequence of neutrals takes the direction of the\n\
      // surrounding strong text if the text on both sides has the same\n\
      // direction. European and Arabic numbers act as if they were R in\n\
      // terms of their influence on neutrals. Start-of-level-run (sor)\n\
      // and end-of-level-run (eor) are used at level run boundaries.\n\
      // N2. Any remaining neutrals take the embedding direction.\n\
      for (var i = 0; i < len; ++i) {\n\
        if (isNeutral.test(types[i])) {\n\
          for (var end = i + 1; end < len && isNeutral.test(types[end]); ++end) {}\n\
          var before = (i ? types[i-1] : startType) == \"L\";\n\
          var after = (end < len - 1 ? types[end] : startType) == \"L\";\n\
          var replace = before || after ? \"L\" : \"R\";\n\
          for (var j = i; j < end; ++j) types[j] = replace;\n\
          i = end - 1;\n\
        }\n\
      }\n\
\n\
      // Here we depart from the documented algorithm, in order to avoid\n\
      // building up an actual levels array. Since there are only three\n\
      // levels (0, 1, 2) in an implementation that doesn't take\n\
      // explicit embedding into account, we can build up the order on\n\
      // the fly, without following the level-based algorithm.\n\
      var order = [], m;\n\
      for (var i = 0; i < len;) {\n\
        if (countsAsLeft.test(types[i])) {\n\
          var start = i;\n\
          for (++i; i < len && countsAsLeft.test(types[i]); ++i) {}\n\
          order.push({from: start, to: i, level: 0});\n\
        } else {\n\
          var pos = i, at = order.length;\n\
          for (++i; i < len && types[i] != \"L\"; ++i) {}\n\
          for (var j = pos; j < i;) {\n\
            if (countsAsNum.test(types[j])) {\n\
              if (pos < j) order.splice(at, 0, {from: pos, to: j, level: 1});\n\
              var nstart = j;\n\
              for (++j; j < i && countsAsNum.test(types[j]); ++j) {}\n\
              order.splice(at, 0, {from: nstart, to: j, level: 2});\n\
              pos = j;\n\
            } else ++j;\n\
          }\n\
          if (pos < i) order.splice(at, 0, {from: pos, to: i, level: 1});\n\
        }\n\
      }\n\
      if (order[0].level == 1 && (m = str.match(/^\\s+/))) {\n\
        order[0].from = m[0].length;\n\
        order.unshift({from: 0, to: m[0].length, level: 0});\n\
      }\n\
      if (lst(order).level == 1 && (m = str.match(/\\s+$/))) {\n\
        lst(order).to -= m[0].length;\n\
        order.push({from: len - m[0].length, to: len, level: 0});\n\
      }\n\
      if (order[0].level != lst(order).level)\n\
        order.push({from: len, to: len, level: order[0].level});\n\
\n\
      return order;\n\
    };\n\
  })();\n\
\n\
  // THE END\n\
\n\
  CodeMirror.version = \"3.0 B\";\n\
\n\
  return CodeMirror;\n\
})();";
