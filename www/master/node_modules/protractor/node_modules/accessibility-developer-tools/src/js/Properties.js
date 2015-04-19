// Copyright 2012 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

goog.require('axs.utils');

goog.provide('axs.properties');

/**
 * @const
 * @type {string}
 */
axs.properties.TEXT_CONTENT_XPATH = './/text()[normalize-space(.)!=""]/parent::*[name()!="script"]';

/**
 * @param {Element} element
 * @return {Object.<string, Object>}
 */
axs.properties.getFocusProperties = function(element) {
    var focusProperties = {};
    var tabindex = element.getAttribute('tabindex');
    if (tabindex != undefined) {
        focusProperties['tabindex'] = { value: tabindex, valid: true };
    } else {
        if (axs.utils.isElementImplicitlyFocusable(element))
            focusProperties['implicitlyFocusable'] = { value: true, valid: true };
    }
    if (Object.keys(focusProperties).length == 0)
        return null;
    var transparent = axs.utils.elementIsTransparent(element);
    var zeroArea = axs.utils.elementHasZeroArea(element);
    var outsideScrollArea = axs.utils.elementIsOutsideScrollArea(element);
    var overlappingElements = axs.utils.overlappingElements(element);
    if (transparent || zeroArea || outsideScrollArea || overlappingElements.length > 0) {
        var hidden = axs.utils.isElementOrAncestorHidden(element);
        var visibleProperties = { value: false,
                                  valid: hidden };
        if (transparent)
            visibleProperties['transparent'] = true;
        if (zeroArea)
            visibleProperties['zeroArea'] = true;
        if (outsideScrollArea)
            visibleProperties['outsideScrollArea'] = true;
        if (overlappingElements && overlappingElements.length > 0)
            visibleProperties['overlappingElements'] = overlappingElements;
        var hiddenProperties = { value: hidden, valid: hidden };
        if (hidden)
            hiddenProperties['reason'] = axs.properties.getHiddenReason(element);
        visibleProperties['hidden'] = hiddenProperties;
        focusProperties['visible'] = visibleProperties;
    } else {
        focusProperties['visible'] = { value: true, valid: true };
    }

    return focusProperties;
};

axs.properties.getHiddenReason = function(element) {
    if (!element || !(element instanceof element.ownerDocument.defaultView.HTMLElement))
      return null;

    if (element.hasAttribute('chromevoxignoreariahidden'))
        var chromevoxignoreariahidden = true;

    var style = window.getComputedStyle(element, null);
    if (style.display == 'none')
        return { 'property': 'display: none',
                 'on': element };

    if (style.visibility == 'hidden')
        return { 'property': 'visibility: hidden',
                 'on': element };

    if (element.hasAttribute('aria-hidden') &&
        element.getAttribute('aria-hidden').toLowerCase() == 'true') {
        if (!chromevoxignoreariahidden)
            return { 'property': 'aria-hidden',
                     'on': element };
    }

    return axs.properties.getHiddenReason(axs.utils.parentElement(element));
};


/**
 * @param {Element} element
 * @return {Object.<string, Object>}
 */
axs.properties.getColorProperties = function(element) {
    var colorProperties = {};
    var contrastRatioProperties =
        axs.properties.getContrastRatioProperties(element);
    if (contrastRatioProperties)
        colorProperties['contrastRatio'] = contrastRatioProperties;
    if (Object.keys(colorProperties).length == 0)
        return null;
    return colorProperties;
};

/**
 * Determines whether the given element has a text node as a direct descendant.
 * @param {Element} element
 * @return {boolean}
 */
axs.properties.hasDirectTextDescendant = function(element) {
    var ownerDocument;
    if (element.nodeType == Node.DOCUMENT_NODE)
        ownerDocument = element;
    else
        ownerDocument = element.ownerDocument;
    if (ownerDocument.evaluate) {
        return hasDirectTextDescendantXpath();
    }
    else {  // IE
        return hasDirectTextDescendantTreeWalker();
    }

    /**
     * Determines whether element has a text node as a direct descendant.
     * This method uses XPath on HTML DOM which is not universally supported.
     */
    function hasDirectTextDescendantXpath() {
        var selectorResults = ownerDocument.evaluate(axs.properties.TEXT_CONTENT_XPATH,
                                                     element,
                                                     null,
                                                     XPathResult.ANY_TYPE,
                                                     null);
        for (var resultElement = selectorResults.iterateNext();
             resultElement != null;
             resultElement = selectorResults.iterateNext()) {
            if (resultElement !== element)
                continue;
            return true;
        }
        return false;
    }

    /**
     * Determines whether element has a text node as a direct descendant.
     * This method uses TreeWalker as a fallback (at time of writing no version
     * of IE (including IE11) supports XPath in the HTML DOM).
     */
    function hasDirectTextDescendantTreeWalker() {
        var treeWalker = ownerDocument.createTreeWalker(element,
                                                        NodeFilter.SHOW_TEXT,
                                                        null,
                                                        false);
        while (treeWalker.nextNode()) {
            var resultElement = treeWalker.currentNode;
            var parent = resultElement.parentNode;
            var tagName = parent.tagName.toLowerCase();
            var value = resultElement.nodeValue.trim();
            if (value && tagName !== 'script' && element !== resultElement)
                return true;
        }
        return false;
    }
};

/**
 * @param {Element} element
 * @return {Object.<string, Object>}
 */
axs.properties.getContrastRatioProperties = function(element) {
    if (!axs.properties.hasDirectTextDescendant(element))
        return null;

    var contrastRatioProperties = {};
    var style = window.getComputedStyle(element, null);
    var bgColor = axs.utils.getBgColor(style, element);
    if (!bgColor)
        return null;

    contrastRatioProperties['backgroundColor'] = axs.utils.colorToString(bgColor);
    var fgColor = axs.utils.getFgColor(style, element, bgColor);
    contrastRatioProperties['foregroundColor'] = axs.utils.colorToString(fgColor);
    var value = axs.utils.getContrastRatioForElementWithComputedStyle(style, element);
    if (!value)
        return null;
    contrastRatioProperties['value'] = value.toFixed(2);
    if (axs.utils.isLowContrast(value, style))
        contrastRatioProperties['alert'] = true;
    var suggestedColors = axs.utils.suggestColors(bgColor, fgColor, value, style);
    if (suggestedColors && Object.keys(suggestedColors).length)
        contrastRatioProperties['suggestedColors'] = suggestedColors;
    return contrastRatioProperties;
};

/**
 * @param {Node} node
 * @param {!Object} textAlternatives The properties object to fill in
 * @param {boolean=} opt_recursive Whether this is a recursive call or not
 * @param {boolean=} opt_force Whether to return text alternatives for this
 *     element regardless of its hidden state.
 * @return {?string} The calculated text alternative for the given element
 */
axs.properties.findTextAlternatives = function(node, textAlternatives, opt_recursive, opt_force) {
    var recursive = opt_recursive || false;

    /** @type {Element} */ var element = axs.utils.asElement(node);
    if (!element)
        return null;

    // 1. Skip hidden elements unless the author specifies to use them via an aria-labelledby or
    // aria-describedby being used in the current computation.
    if (!recursive && !opt_force && axs.utils.isElementOrAncestorHidden(element))
        return null;

    // if this is a text node, just return text content.
    if (node.nodeType == Node.TEXT_NODE) {
        var textContentValue = {};
        textContentValue.type = 'text';
        textContentValue.text = node.textContent;
        textContentValue.lastWord = axs.properties.getLastWord(textContentValue.text);
        textAlternatives['content'] = textContentValue;

        return node.textContent;
    }

    var computedName = null;

    if (!recursive) {
        // 2A. The aria-labelledby attribute takes precedence as the element's text alternative
        // unless this computation is already occurring as the result of a recursive aria-labelledby
        // declaration.
        computedName = axs.properties.getTextFromAriaLabelledby(element, textAlternatives);
    }

    // 2A. If aria-labelledby is empty or undefined, the aria-label attribute, which defines an
    // explicit text string, is used.
    if (element.hasAttribute('aria-label')) {
        var ariaLabelValue = {};
        ariaLabelValue.type = 'text';
        ariaLabelValue.text = element.getAttribute('aria-label');
        ariaLabelValue.lastWord = axs.properties.getLastWord(ariaLabelValue.text);
        if (computedName)
            ariaLabelValue.unused = true;
        else if (!(recursive && axs.utils.elementIsHtmlControl(element)))
            computedName = ariaLabelValue.text;
        textAlternatives['ariaLabel'] = ariaLabelValue;
    }

    // 2A. If aria-labelledby and aria-label are both empty or undefined, and if the element is not
    // marked as presentational (role="presentation", check for the presence of an equivalent host
    // language attribute or element for associating a label, and use those mechanisms to determine
    // a text alternative.
    if (!element.hasAttribute('role') || element.getAttribute('role') != 'presentation') {
        computedName = axs.properties.getTextFromHostLanguageAttributes(element,
                                                                        textAlternatives,
                                                                        computedName,
                                                                        recursive);
    }

    // 2B (HTML version).
    if (recursive && axs.utils.elementIsHtmlControl(element)) {
        var defaultView = element.ownerDocument.defaultView;

        // include the value of the embedded control as part of the text alternative in the
        // following manner:
        if (element instanceof defaultView.HTMLInputElement) {
            // If the embedded control is a text field, use its value.
            var inputElement = /** @type {HTMLInputElement} */ (element);
            if (inputElement.type == 'text') {
                if (inputElement.value && inputElement.value.length > 0)
                    textAlternatives['controlValue'] = { 'text': inputElement.value };
            }
            // If the embedded control is a range (e.g. a spinbutton or slider), use the value of the
            // aria-valuetext attribute if available, or otherwise the value of the aria-valuenow
            // attribute.
            if (inputElement.type == 'range')
                textAlternatives['controlValue'] = { 'text': inputElement.value };
        }
        // If the embedded control is a menu, use the text alternative of the chosen menu item.
        // If the embedded control is a select or combobox, use the chosen option.
        if (element instanceof defaultView.HTMLSelectElement) {
            var inputElement = /** @type {HTMLSelectElement} */ (element);
            textAlternatives['controlValue'] = { 'text': inputElement.value };
        }

        if (textAlternatives['controlValue']) {
            var controlValue = textAlternatives['controlValue'];
            if (computedName)
                controlValue.unused = true;
            else
                computedName = controlValue.text;
        }
    }

    // 2B (ARIA version).
    if (recursive && axs.utils.elementIsAriaWidget(element)) {
        var role = element.getAttribute('role');
        // If the embedded control is a text field, use its value.
        if (role == 'textbox') {
            if (element.textContent && element.textContent.length > 0)
                textAlternatives['controlValue'] = { 'text': element.textContent };
        }
        // If the embedded control is a range (e.g. a spinbutton or slider), use the value of the
        // aria-valuetext attribute if available, or otherwise the value of the aria-valuenow
        // attribute.
        if (role == 'slider' || role == 'spinbutton') {
            if (element.hasAttribute('aria-valuetext'))
                textAlternatives['controlValue'] = { 'text': element.getAttribute('aria-valuetext') };
            else if (element.hasAttribute('aria-valuenow'))
                textAlternatives['controlValue'] = { 'value': element.getAttribute('aria-valuenow'),
                                                     'text': '' + element.getAttribute('aria-valuenow')};
        }
        // If the embedded control is a menu, use the text alternative of the chosen menu item.
        if (role == 'menu') {
            var menuitems = element.querySelectorAll('[role=menuitemcheckbox], [role=menuitemradio]');
            var selectedMenuitems = [];
            for (var i = 0; i < menuitems.length; i++) {
                if (menuitems[i].getAttribute('aria-checked') == 'true')
                    selectedMenuitems.push(menuitems[i]);
            }
            if (selectedMenuitems.length > 0) {
                var selectedMenuText = '';
                for (var i = 0; i < selectedMenuitems.length; i++) {
                    selectedMenuText += axs.properties.findTextAlternatives(selectedMenuitems[i], {}, true);
                    if (i < selectedMenuitems.length - 1)
                        selectedMenuText += ', ';
                }
                textAlternatives['controlValue'] = { 'text': selectedMenuText };
            }
        }
        // If the embedded control is a select or combobox, use the chosen option.
        if (role == 'combobox' || role == 'select') {
            // TODO
            textAlternatives['controlValue'] = { 'text': 'TODO' };
        }

        if (textAlternatives['controlValue']) {
            var controlValue = textAlternatives['controlValue'];
            if (computedName)
                controlValue.unused = true;
            else
                computedName = controlValue.text;
        }
    }

    // 2C. Otherwise, if the attributes checked in rules A and B didn't provide results, text is
    // collected from descendant content if the current element's role allows "Name From: contents."
    var hasRole = element.hasAttribute('role');
    var canGetNameFromContents = true;
    if (hasRole) {
        var roleName = element.getAttribute('role');
        // if element has a role, check that it allows "Name From: contents"
        var role = axs.constants.ARIA_ROLES[roleName];
        if (role && (!role.namefrom || role.namefrom.indexOf('contents') < 0))
            canGetNameFromContents = false;
    }
    var textFromContent = axs.properties.getTextFromDescendantContent(element);
    if (textFromContent && canGetNameFromContents) {
        var textFromContentValue = {};
        textFromContentValue.type = 'text';
        textFromContentValue.text = textFromContent;
        textFromContentValue.lastWord = axs.properties.getLastWord(textFromContentValue.text);
        if (computedName)
            textFromContentValue.unused = true;
        else
            computedName = textFromContent;
        textAlternatives['content'] = textFromContentValue;
    }

    // 2D. The last resort is to use text from a tooltip attribute (such as the title attribute in
    // HTML). This is used only if nothing else, including subtree content, has provided results.
    if (element.hasAttribute('title')) {
        var titleValue = {};
        titleValue.type = 'string';
        titleValue.valid = true;
        titleValue.text =  element.getAttribute('title');
        titleValue.lastWord = axs.properties.getLastWord(titleValue.lastWord);
        if (computedName)
            titleValue.unused = true;
        else
            computedName = titleValue.text;
        textAlternatives['title'] = titleValue;
    }

    if (Object.keys(textAlternatives).length == 0 && computedName == null)
        return null;

    return computedName;
};

/**
 * @param {Element} element
 * @return {?string}
 */
axs.properties.getTextFromDescendantContent = function(element) {
    var children = element.childNodes;
    var childrenTextContent = [];
    for (var i = 0; i < children.length; i++) {
        var childTextContent = axs.properties.findTextAlternatives(children[i], {}, true);
        if (childTextContent)
            childrenTextContent.push(childTextContent.trim());
    }
    if (childrenTextContent.length) {
        var result = '';
        // Empty children are allowed, but collapse all of them
        for (var i = 0; i < childrenTextContent.length; i++)
            result = [result, childrenTextContent[i]].join(' ').trim();
        return result;
    }
    return null;
};

/**
 * @param {Element} element
 * @param {Object} textAlternatives
 * @return {?string}
 */
axs.properties.getTextFromAriaLabelledby = function(element, textAlternatives) {
    var computedName = null;
    if (!element.hasAttribute('aria-labelledby'))
        return computedName;

    var labelledbyAttr = element.getAttribute('aria-labelledby');
    var labelledbyIds = labelledbyAttr.split(/\s+/);
    var labelledbyValue = {};
    labelledbyValue.valid = true;
    var labelledbyText = [];
    var labelledbyValues = [];
    for (var i = 0; i < labelledbyIds.length; i++) {
        var labelledby = {};
        labelledby.type = 'element';
        var labelledbyId = labelledbyIds[i];
        labelledby.value = labelledbyId;
        var labelledbyElement = document.getElementById(labelledbyId);
        if (!labelledbyElement) {
            labelledby.valid = false;
            labelledbyValue.valid = false;
            labelledby.errorMessage = { 'messageKey': 'noElementWithId', 'args': [labelledbyId] };
        } else {
            labelledby.valid = true;
            labelledby.text = axs.properties.findTextAlternatives(labelledbyElement, {}, true);
            labelledby.lastWord = axs.properties.getLastWord(labelledby.text);
            labelledbyText.push(labelledbyElement.textContent.trim());
            labelledby.element = labelledbyElement;
        }
        labelledbyValues.push(labelledby);
    }
    if (labelledbyValues.length > 0) {
        labelledbyValues[labelledbyValues.length - 1].last = true;
        labelledbyValue.values = labelledbyValues;
        labelledbyValue.text = labelledbyText.join(' ');
        labelledbyValue.lastWord = axs.properties.getLastWord(labelledbyValue.text);
        computedName = labelledbyValue.text;
        textAlternatives['ariaLabelledby'] = labelledbyValue;
    }

    return computedName;
};

axs.properties.getTextFromHostLanguageAttributes = function(element,
                                                            textAlternatives,
                                                            existingComputedname,
                                                            recursive) {
    var computedName = existingComputedname;
    if (axs.browserUtils.matchSelector(element, 'img')) {
        if (element.hasAttribute('alt')) {
            var altValue = {};
            altValue.type = 'string';
            altValue.valid = true;
            altValue.text =  element.getAttribute('alt');
            if (computedName)
                altValue.unused = true;
            else
                computedName = altValue.text;
            textAlternatives['alt'] = altValue;
        } else {
            var altValue = {};
            altValue.valid = false;
            altValue.errorMessage = 'No alt value provided';
            textAlternatives['alt'] = altValue;
            var src = element.src;
            if (typeof(src) == 'string') {
                var parts = src.split('/');
                var filename = parts.pop();
                var filenameValue = { text: filename }
                textAlternatives['filename'] = filenameValue;
                computedName = filename;
            }
        }
    }

    var controlsSelector = ['input:not([type="hidden"]):not([disabled])',
                            'select:not([disabled])',
                            'textarea:not([disabled])',
                            'button:not([disabled])',
                            'video:not([disabled])'].join(', ');
    if (axs.browserUtils.matchSelector(element, controlsSelector) && !recursive) {
        if (element.hasAttribute('id')) {
            var labelForQuerySelector = 'label[for="' + element.id + '"]';
            var labelsFor = document.querySelectorAll(labelForQuerySelector);
            var labelForValue = {};
            var labelForValues = [];
            var labelForText = [];
            for (var i = 0; i < labelsFor.length; i++) {
                var labelFor = {};
                labelFor.type = 'element';
                var label = labelsFor[i];
                var labelText = axs.properties.findTextAlternatives(label, {}, true);
                if (labelText && labelText.trim().length > 0) {
                    labelFor.text = labelText.trim();
                    labelForText.push(labelText.trim());
                }
                labelFor.element = label;
                labelForValues.push(labelFor);
            }
            if (labelForValues.length > 0) {
                labelForValues[labelForValues.length - 1].last = true;
                labelForValue.values = labelForValues;
                labelForValue.text = labelForText.join(' ');
                labelForValue.lastWord = axs.properties.getLastWord(labelForValue.text);
                if (computedName)
                    labelForValue.unused = true;
                else
                    computedName = labelForValue.text;
                textAlternatives['labelFor'] = labelForValue;
            }
        }

        var parent = axs.utils.parentElement(element);
        var labelWrappedValue = {};
        while (parent) {
            if (parent.tagName.toLowerCase() == 'label') {
                var parentLabel = /** @type {HTMLLabelElement} */ (parent);
                if (parentLabel.control == element) {
                    labelWrappedValue.type = 'element';
                    labelWrappedValue.text = axs.properties.findTextAlternatives(parentLabel, {}, true);
                    labelWrappedValue.lastWord = axs.properties.getLastWord(labelWrappedValue.text);
                    labelWrappedValue.element = parentLabel;
                    break;
                }
            }
            parent = axs.utils.parentElement(parent);
        }
        if (labelWrappedValue.text) {
            if (computedName)
                labelWrappedValue.unused = true;
            else
                computedName = labelWrappedValue.text;
            textAlternatives['labelWrapped'] = labelWrappedValue;
        }
        if (!Object.keys(textAlternatives).length)
            textAlternatives['noLabel'] = true;
    }
    return computedName;
};

/**
 * @param {?string} text
 * @return {?string}
 */
axs.properties.getLastWord = function(text) {
    if (!text)
        return null;

    // TODO: this makes a lot of assumptions.
    var lastSpace = text.lastIndexOf(' ') + 1;
    var MAXLENGTH = 10;
    var cutoff = text.length - MAXLENGTH;
    var wordStart = lastSpace > cutoff ? lastSpace : cutoff;
    return text.substring(wordStart);
};

/**
 * @param {Node} node
 * @return {Object}
 */
axs.properties.getTextProperties = function(node) {
    var textProperties = {};
    var computedName = axs.properties.findTextAlternatives(node, textProperties, false, true);

    if (Object.keys(textProperties).length == 0) {
        if (!computedName)
            return null;
        textProperties.hasProperties = false;
    } else {
        textProperties.hasProperties = true;
    }

    textProperties.computedText = computedName;
    textProperties.lastWord = axs.properties.getLastWord(computedName);
    return textProperties;
};

/**
 * @param {Element} element
 * @return {Object}
 */
axs.properties.getAriaProperties = function(element) {
    var ariaProperties = {};
    var statesAndProperties = axs.properties.getGlobalAriaProperties(element);

    for (var property in axs.constants.ARIA_PROPERTIES) {
        var attributeName = 'aria-' + property;
        if (element.hasAttribute(attributeName)) {
            var propertyValue = element.getAttribute(attributeName);
            statesAndProperties[attributeName] =
                axs.utils.getAriaPropertyValue(attributeName, propertyValue, element);
        }
    }
    if (Object.keys(statesAndProperties).length > 0)
        ariaProperties['properties'] = axs.utils.values(statesAndProperties);

    var roles = axs.utils.getRoles(element);
    if (!roles) {
        if (Object.keys(ariaProperties).length)
            return ariaProperties;
        else
            return null;
    }
    ariaProperties['roles'] = roles;
    if (!roles.valid || !roles['roles'])
        return ariaProperties;

    var roleDetails = roles['roles'];
    for (var i = 0; i < roleDetails.length; i++) {
        var role = roleDetails[i];
        if (!role.details || !role.details.propertiesSet)
            continue;
        for (var property in role.details.propertiesSet) {
            if (property in statesAndProperties)
                continue;
            if (element.hasAttribute(property)) {
                var propertyValue = element.getAttribute(property);
                statesAndProperties[property] =
                    axs.utils.getAriaPropertyValue(property, propertyValue, element);
                if ('values' in statesAndProperties[property]) {
                    var values = statesAndProperties[property].values;
                    values[values.length - 1].isLast = true;
                }
            } else if (role.details.requiredPropertiesSet[property]) {
                statesAndProperties[property] =
                    { 'name': property, 'valid': false, 'reason': 'Required property not set' };
            }
        }
    }
    if (Object.keys(statesAndProperties).length > 0)
        ariaProperties['properties'] = axs.utils.values(statesAndProperties);
    if (Object.keys(ariaProperties).length > 0)
        return ariaProperties;
    return null;
};

/**
 * Gets the ARIA properties which apply to all elements, not just elements with ARIA roles.
 * @param {Element} element
 * @return {!Object}
 */
axs.properties.getGlobalAriaProperties = function(element) {
    var globalProperties = {};
    for (var i = 0; i < axs.constants.GLOBAL_PROPERTIES.length; i++) {
        var property = axs.constants.GLOBAL_PROPERTIES[i];
        if (element.hasAttribute(property)) {
            var propertyValue = element.getAttribute(property);
            globalProperties[property] =
                axs.utils.getAriaPropertyValue(property, propertyValue, element);
        }
    }
    return globalProperties;
};

/**
 * @param {Element} element
 * @return {Object.<string, Object>}
 */
axs.properties.getVideoProperties = function(element) {
    var videoSelector = 'video';
    if (!axs.browserUtils.matchSelector(element, videoSelector))
        return null;
    var videoProperties = {};
    videoProperties['captionTracks'] = axs.properties.getTrackElements(element, 'captions');
    videoProperties['descriptionTracks'] = axs.properties.getTrackElements(element, 'descriptions');
    videoProperties['chapterTracks'] = axs.properties.getTrackElements(element, 'chapters');
    // error if no text alternatives?
    return videoProperties;
};

/**
 * @param {Element} element
 * @param {string} kind
 * @return {Object}
 */
axs.properties.getTrackElements = function(element, kind) {
    // error if resource is not available
    var trackElements = element.querySelectorAll('track[kind=' + kind + ']');
    var result = {};
    if (!trackElements.length) {
        result.valid = false;
        result.reason = { 'messageKey': 'noTracksProvided', 'args': [[kind]] };
        return result;
    }
    result.valid = true;
    var values = [];
    for (var i = 0; i < trackElements.length; i++) {
        var trackElement = {};
        var src = trackElements[i].getAttribute('src');
        var srcLang = trackElements[i].getAttribute('srcLang');
        var label = trackElements[i].getAttribute('label');
        if (!src) {
            trackElement.valid = false;
            trackElement.reason = { 'messageKey': 'noSrcProvided' };
        } else {
            trackElement.valid = true;
            trackElement.src = src;
        }
        var name = '';
        if (label) {
            name += label;
            if (srcLang)
                name += ' ';
        }
        if (srcLang)
            name += '(' + srcLang + ')';
        if (name == '')
            name = '[' + { 'messageKey': 'unnamed' } + ']';
        trackElement.name = name;
        values.push(trackElement);
    }
    result.values = values;
    return result;
};

/**
 * @param {Node} node
 * @return {Object.<string, Object>}
 */
axs.properties.getAllProperties = function(node) {
    /** @type {Element} */ var element = axs.utils.asElement(node);
    if (!element)
        return {};

    var allProperties = {};
    allProperties['ariaProperties'] = axs.properties.getAriaProperties(element);
    allProperties['colorProperties'] = axs.properties.getColorProperties(element);
    allProperties['focusProperties'] = axs.properties.getFocusProperties(element);
    allProperties['textProperties'] = axs.properties.getTextProperties(node);
    allProperties['videoProperties'] = axs.properties.getVideoProperties(element);
    return allProperties;
};
