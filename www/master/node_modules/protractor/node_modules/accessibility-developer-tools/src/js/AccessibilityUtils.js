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

goog.require('axs.constants');
goog.provide('axs.utils');
goog.provide('axs.utils.Color');

/**
 * @const
 * @type {string}
 */
axs.utils.FOCUSABLE_ELEMENTS_SELECTOR =
    'input:not([type=hidden]):not([disabled]),' +
    'select:not([disabled]),' +
    'textarea:not([disabled]),' +
    'button:not([disabled]),' +
    'a[href],' +
    'iframe,' +
    '[tabindex]';

/**
 * @constructor
 * @param {number} red
 * @param {number} green
 * @param {number} blue
 * @param {number} alpha
 */
axs.utils.Color = function(red, green, blue, alpha) {
    /** @type {number} */
    this.red = red;

    /** @type {number} */
    this.green = green;

    /** @type {number} */
    this.blue = blue;

    /** @type {number} */
    this.alpha = alpha;
};

/**
 * Calculate the contrast ratio between the two given colors. Returns the ratio
 * to 1, for example for two two colors with a contrast ratio of 21:1, this
 * function will return 21.
 * @param {axs.utils.Color} fgColor
 * @param {axs.utils.Color} bgColor
 * @return {?number}
 */
axs.utils.calculateContrastRatio = function(fgColor, bgColor) {
    if (!fgColor || !bgColor)
        return null;

    if (fgColor.alpha < 1)
        fgColor = axs.utils.flattenColors(fgColor, bgColor);

    var fgLuminance = axs.utils.calculateLuminance(fgColor);
    var bgLuminance = axs.utils.calculateLuminance(bgColor);
    var contrastRatio = (Math.max(fgLuminance, bgLuminance) + 0.05) /
        (Math.min(fgLuminance, bgLuminance) + 0.05);
    return contrastRatio;
};

axs.utils.luminanceRatio = function(luminance1, luminance2) {
    return (Math.max(luminance1, luminance2) + 0.05) /
        (Math.min(luminance1, luminance2) + 0.05);
};

/**
 * Returns the nearest ancestor which is an Element.
 * @param {Node} node
 * @return {Element}
 */
axs.utils.parentElement = function(node) {
    if (!node)
        return null;
    if (node.nodeType == Node.DOCUMENT_FRAGMENT_NODE)
        return node.host;

    var parentElement = node.parentElement;
    if (parentElement)
        return parentElement;

    var parentNode = node.parentNode;
    if (!parentNode)
        return null;

    switch (parentNode.nodeType) {
    case Node.ELEMENT_NODE:
        return /** @type {Element} */ (parentNode);
    case Node.DOCUMENT_FRAGMENT_NODE:
        return parentNode.host;
    default:
        return null;
    }
};

/**
 * Return the corresponding element for the given node.
 * @param {Node} node
 * @return {Element}
 * @suppress {checkTypes}
 */
axs.utils.asElement = function(node) {
    /** @type {Element} */ var element;
    switch (node.nodeType) {
    case Node.COMMENT_NODE:
        return null;  // Skip comments
    case Node.ELEMENT_NODE:
        element = /** (@type {Element}) */ node;
        if (element.tagName.toLowerCase() == 'script')
            return null;  // Skip script elements
        break;
    case Node.TEXT_NODE:
        element = axs.utils.parentElement(node);
        break;
    default:
        console.warn('Unhandled node type: ', node.nodeType);
        return null;
    }
    return element;
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementIsTransparent = function(element) {
    return element.style.opacity == '0';
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementHasZeroArea = function(element) {
    var rect = element.getBoundingClientRect();
    var width = rect.right - rect.left;
    var height = rect.top - rect.bottom;
    if (!width || !height)
        return true;
    return false;
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementIsOutsideScrollArea = function(element) {
    var parent = axs.utils.parentElement(element);

    var defaultView = element.ownerDocument.defaultView;
    while (parent != defaultView.document.body) {
        if (axs.utils.isClippedBy(element, parent))
            return true;

        if (axs.utils.canScrollTo(element, parent) && !axs.utils.elementIsOutsideScrollArea(parent))
            return false;

        parent = axs.utils.parentElement(parent);
    }

    return !axs.utils.canScrollTo(element, defaultView.document.body);
};

/**
 * Checks whether it's possible to scroll to the given element within the given container.
 * Assumes that |container| is an ancestor of |element|.
 * If |container| cannot be scrolled, returns True if the element is within its bounding client
 * rect.
 * @param {Element} element
 * @param {Element} container
 * @return {boolean} True iff it's possible to scroll to |element| within |container|.
 */
axs.utils.canScrollTo = function(element, container) {
    var rect = element.getBoundingClientRect();
    var containerRect = container.getBoundingClientRect();
    var containerTop = containerRect.top;
    var containerLeft = containerRect.left;
    var containerScrollArea =
        { top: containerTop - container.scrollTop,
          bottom: containerTop - container.scrollTop + container.scrollHeight,
          left: containerLeft - container.scrollLeft,
          right: containerLeft - container.scrollLeft + container.scrollWidth };

    if (rect.right < containerScrollArea.left || rect.bottom < containerScrollArea.top ||
            rect.left > containerScrollArea.right || rect.top > containerScrollArea.bottom) {
        return false;
    }

    var defaultView = element.ownerDocument.defaultView;
    var style = defaultView.getComputedStyle(container);

    if (rect.left > containerRect.right || rect.top > containerRect.bottom) {
        return (style.overflow == 'scroll' || style.overflow == 'auto' ||
                container instanceof defaultView.HTMLBodyElement);
    }

    return true;
};

/**
 * Checks whether the given element is clipped by the given container.
 * Assumes that |container| is an ancestor of |element|.
 * @param {Element} element
 * @param {Element} container
 * @return {boolean} True iff |element| is clipped by |container|.
 */
axs.utils.isClippedBy = function(element, container) {
    var rect = element.getBoundingClientRect();
    var containerRect = container.getBoundingClientRect();
    var containerTop = containerRect.top;
    var containerLeft = containerRect.left;
    var containerScrollArea =
        { top: containerTop - container.scrollTop,
          bottom: containerTop - container.scrollTop + container.scrollHeight,
          left: containerLeft - container.scrollLeft,
          right: containerLeft - container.scrollLeft + container.scrollWidth };

    var defaultView = element.ownerDocument.defaultView;
    var style = defaultView.getComputedStyle(container);

    if ((rect.right < containerRect.left || rect.bottom < containerRect.top ||
             rect.left > containerRect.right || rect.top > containerRect.bottom) &&
             style.overflow == 'hidden') {
        return true;
    }

    if (rect.right < containerScrollArea.left || rect.bottom < containerScrollArea.top)
        return (style.overflow != 'visible');

    return false;
};

/**
 * @param {Node} ancestor A potential ancestor of |node|.
 * @param {Node} node
 * @return {boolean} true if |ancestor| is an ancestor of |node| (including
 *     |ancestor| === |node|).
 */
axs.utils.isAncestor = function(ancestor, node) {
    if (node == null)
        return false;
    if (node === ancestor)
        return true;

    return axs.utils.isAncestor(ancestor, node.parentNode);
};

/**
 * @param {Element} element
 * @return {Array.<Element>} An array of any non-transparent elements which
 *     overlap the given element.
 */
axs.utils.overlappingElements = function(element) {
    if (axs.utils.elementHasZeroArea(element))
        return null;

    var overlappingElements = [];
    var clientRects = element.getClientRects();
    for (var i = 0; i < clientRects.length; i++) {
        var rect = clientRects[i];
        var center_x = (rect.left + rect.right) / 2;
        var center_y = (rect.top + rect.bottom) / 2;
        var elementAtPoint = document.elementFromPoint(center_x, center_y);

        if (elementAtPoint == null || elementAtPoint == element ||
            axs.utils.isAncestor(elementAtPoint, element) ||
            axs.utils.isAncestor(element, elementAtPoint)) {
            continue;
        }

        var overlappingElementStyle = window.getComputedStyle(elementAtPoint, null);
        if (!overlappingElementStyle)
            continue;

        var overlappingElementBg = axs.utils.getBgColor(overlappingElementStyle,
                                                        elementAtPoint);
        if (overlappingElementBg && overlappingElementBg.alpha > 0 &&
            overlappingElements.indexOf(elementAtPoint) < 0) {
            overlappingElements.push(elementAtPoint);
        }
    }

    return overlappingElements;
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementIsHtmlControl = function(element) {
    var defaultView = element.ownerDocument.defaultView;

    // HTML control
    if (element instanceof defaultView.HTMLButtonElement)
        return true;
    if (element instanceof defaultView.HTMLInputElement)
        return true;
    if (element instanceof defaultView.HTMLSelectElement)
        return true;
    if (element instanceof defaultView.HTMLTextAreaElement)
        return true;

    return false;
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementIsAriaWidget = function(element) {
    if (element.hasAttribute('role')) {
        var roleValue = element.getAttribute('role');
        // TODO is this correct?
        if (roleValue) {
            var role = axs.constants.ARIA_ROLES[roleValue];
            if (role && 'widget' in role['allParentRolesSet'])
                return true;
        }
    }
    return false;
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.elementIsVisible = function(element) {
    if (axs.utils.elementIsTransparent(element))
        return false;
    if (axs.utils.elementHasZeroArea(element))
        return false;
    if (axs.utils.elementIsOutsideScrollArea(element))
        return false;

    var overlappingElements = axs.utils.overlappingElements(element);
    if (overlappingElements.length)
        return false;

    return true;
};

/**
 * @param {CSSStyleDeclaration} style
 * @return {boolean}
 */
axs.utils.isLargeFont = function(style) {
    var fontSize = style.fontSize;
    var bold = style.fontWeight == 'bold';
    var matches = fontSize.match(/(\d+)px/);
    if (matches) {
        var fontSizePx = parseInt(matches[1], 10);
        var bodyStyle = window.getComputedStyle(document.body, null);
        var bodyFontSize = bodyStyle.fontSize;
        matches = bodyFontSize.match(/(\d+)px/);
        if (matches) {
            var bodyFontSizePx = parseInt(matches[1], 10);
            var boldLarge = bodyFontSizePx * 1.2;
            var large = bodyFontSizePx * 1.5;
        } else {
            var boldLarge = 19.2;
            var large = 24;
        }
        return (bold && fontSizePx >= boldLarge || fontSizePx >= large);
    }
    matches = fontSize.match(/(\d+)em/);
    if (matches) {
        var fontSizeEm = parseInt(matches[1], 10);
        if (bold && fontSizeEm >= 1.2 || fontSizeEm >= 1.5)
            return true;
        return false;
    }
    matches = fontSize.match(/(\d+)%/);
    if (matches) {
        var fontSizePercent = parseInt(matches[1], 10);
        if (bold && fontSizePercent >= 120 || fontSizePercent >= 150)
            return true;
        return false;
    }
    matches = fontSize.match(/(\d+)pt/);
    if (matches) {
        var fontSizePt = parseInt(matches[1], 10);
        if (bold && fontSizePt >= 14 || fontSizePt >= 18)
            return true;
        return false;
    }
    return false;
};

/**
 * @param {CSSStyleDeclaration} style
 * @param {Element} element
 * @return {?axs.utils.Color}
 */
axs.utils.getBgColor = function(style, element) {
    var bgColorString = style.backgroundColor;
    var bgColor = axs.utils.parseColor(bgColorString);
    if (!bgColor)
        return null;

    if (style.opacity < 1)
        bgColor.alpha = bgColor.alpha * style.opacity;

    if (bgColor.alpha < 1) {
        var parentBg = axs.utils.getParentBgColor(element);
        if (parentBg == null)
            return null;

        bgColor = axs.utils.flattenColors(bgColor, parentBg);
    }
    return bgColor;
};

/**
 * Gets the effective background color of the parent of |element|.
 * @param {Element} element
 * @return {?axs.utils.Color}
 */
axs.utils.getParentBgColor = function(element) {
    /** @type {Element} */ var parent = element;
    var bgStack = [];
    var foundSolidColor = null;
    while (parent = axs.utils.parentElement(parent)) {
        var computedStyle = window.getComputedStyle(parent, null);
        if (!computedStyle)
            continue;

        var parentBg = axs.utils.parseColor(computedStyle.backgroundColor);
        if (!parentBg)
            continue;

        if (computedStyle.opacity < 1)
            parentBg.alpha = parentBg.alpha * computedStyle.opacity;

        if (parentBg.alpha == 0)
            continue;

        bgStack.push(parentBg);

        if (parentBg.alpha == 1) {
            foundSolidColor = true;
            break;
        }
    }

    if (!foundSolidColor)
        bgStack.push(new axs.utils.Color(255, 255, 255, 1));

    var bg = bgStack.pop();
    while (bgStack.length) {
        var fg = bgStack.pop();
        bg = axs.utils.flattenColors(fg, bg);
    }
    return bg;
};

/**
 * @param {CSSStyleDeclaration} style
 * @param {Element} element
 * @param {axs.utils.Color} bgColor The background color, which may come from
 *    another element (such as a parent element), for flattening into the
 *    foreground color.
 * @return {?axs.utils.Color}
 */
axs.utils.getFgColor = function(style, element, bgColor) {
    var fgColorString = style.color;
    var fgColor = axs.utils.parseColor(fgColorString);
    if (!fgColor)
        return null;

    if (fgColor.alpha < 1)
        fgColor = axs.utils.flattenColors(fgColor, bgColor);

    if (style.opacity < 1) {
        var parentBg = axs.utils.getParentBgColor(element);
        fgColor.alpha = fgColor.alpha * style.opacity;
        fgColor = axs.utils.flattenColors(fgColor, parentBg);
    }

    return fgColor;
};

/**
 * @param {string} colorString The color string from CSS.
 * @return {?axs.utils.Color}
 */
axs.utils.parseColor = function(colorString) {
    var rgbRegex = /^rgb\((\d+), (\d+), (\d+)\)$/;
    var match = colorString.match(rgbRegex);

    if (match) {
        var r = parseInt(match[1], 10);
        var g = parseInt(match[2], 10);
        var b = parseInt(match[3], 10);
        var a = 1;
        return new axs.utils.Color(r, g, b, a);
    }

    var rgbaRegex = /^rgba\((\d+), (\d+), (\d+), (\d*(\.\d+)?)\)/;
    match = colorString.match(rgbaRegex);
    if (match) {
        var r = parseInt(match[1], 10);
        var g = parseInt(match[2], 10);
        var b = parseInt(match[3], 10);
        var a = parseFloat(match[4]);
        return new axs.utils.Color(r, g, b, a);
    }

    return null;
};

/**
 * @param {number} value The value of a color channel, 0 <= value <= 0xFF
 * @return {string}
 */
axs.utils.colorChannelToString = function(value) {
    value = Math.round(value);
    if (value <= 0xF)
        return '0' + value.toString(16);
    return value.toString(16);
};

/**
 * @param {axs.utils.Color} color
 * @return {string}
 */
axs.utils.colorToString = function(color) {
    if (color.alpha == 1) {
         return '#' + axs.utils.colorChannelToString(color.red) +
         axs.utils.colorChannelToString(color.green) + axs.utils.colorChannelToString(color.blue);
    }
    else
        return 'rgba(' + [color.red, color.green, color.blue, color.alpha].join(',') + ')';
};

axs.utils.luminanceFromContrastRatio = function(luminance, contrast, higher) {
    if (higher) {
        var newLuminance = (luminance + 0.05) * contrast - 0.05;
        return newLuminance;
    } else {
        var newLuminance = (luminance + 0.05) / contrast - 0.05;
        return newLuminance;
    }
};

axs.utils.translateColor = function(ycc, luminance) {
    var oldLuminance = ycc[0];
    if (oldLuminance > luminance)
        var endpoint = 0;
    else
        var endpoint = 1;

    var d = luminance - oldLuminance;
    var scale = d / (endpoint - oldLuminance);

    /** @type {Array.<number>} */ var translatedColor = [ luminance,
                                                          ycc[1] - ycc[1] * scale,
                                                          ycc[2] - ycc[2] * scale ];
    var rgb = axs.utils.fromYCC(translatedColor);
    return rgb;
};

/**
 * @param {axs.utils.Color} bgColor
 * @param {axs.utils.Color} fgColor
 * @param {number} contrastRatio
 * @param {CSSStyleDeclaration} style
 * @return {Object}
 */
axs.utils.suggestColors = function(bgColor, fgColor, contrastRatio, style) {
    if (!axs.utils.isLowContrast(contrastRatio, style, true))
        return null;
    var colors = {};
    var bgLuminance = axs.utils.calculateLuminance(bgColor);
    var fgLuminance = axs.utils.calculateLuminance(fgColor);

    var levelAAContrast = axs.utils.isLargeFont(style) ? 3.0 : 4.5;
    var levelAAAContrast = axs.utils.isLargeFont(style) ? 4.5 : 7.0;
    var fgLuminanceIsHigher = fgLuminance > bgLuminance;
    var desiredFgLuminanceAA = axs.utils.luminanceFromContrastRatio(bgLuminance, levelAAContrast + 0.02, fgLuminanceIsHigher);
    var desiredFgLuminanceAAA = axs.utils.luminanceFromContrastRatio(bgLuminance, levelAAAContrast + 0.02, fgLuminanceIsHigher);
    var fgYCC = axs.utils.toYCC(fgColor);

    if (axs.utils.isLowContrast(contrastRatio, style, false) &&
        desiredFgLuminanceAA <= 1 && desiredFgLuminanceAA >= 0) {
        var newFgColorAA = axs.utils.translateColor(fgYCC, desiredFgLuminanceAA);
        var newContrastRatioAA = axs.utils.calculateContrastRatio(newFgColorAA, bgColor);
        var suggestedColorsAA = {};
        suggestedColorsAA['fg'] = axs.utils.colorToString(newFgColorAA);
        suggestedColorsAA['bg'] = axs.utils.colorToString(bgColor);
        suggestedColorsAA['contrast'] = newContrastRatioAA.toFixed(2);
        colors['AA'] = suggestedColorsAA;
    }
    if (axs.utils.isLowContrast(contrastRatio, style, true) &&
        desiredFgLuminanceAAA <= 1 && desiredFgLuminanceAAA >= 0) {
        var newFgColorAAA = axs.utils.translateColor(fgYCC, desiredFgLuminanceAAA);
        var newContrastRatioAAA = axs.utils.calculateContrastRatio(newFgColorAAA, bgColor);
        var suggestedColorsAAA = {};
        suggestedColorsAAA['fg'] = axs.utils.colorToString(newFgColorAAA);
        suggestedColorsAAA['bg'] = axs.utils.colorToString(bgColor);
        suggestedColorsAAA['contrast'] = newContrastRatioAAA.toFixed(2);
        colors['AAA'] = suggestedColorsAAA;
    }
    var desiredBgLuminanceAA = axs.utils.luminanceFromContrastRatio(fgLuminance, levelAAContrast + 0.02, !fgLuminanceIsHigher);
    var desiredBgLuminanceAAA = axs.utils.luminanceFromContrastRatio(fgLuminance, levelAAAContrast + 0.02, !fgLuminanceIsHigher);
    var bgYCC = axs.utils.toYCC(bgColor);

    if (!('AA' in colors) && axs.utils.isLowContrast(contrastRatio, style, false) &&
        desiredBgLuminanceAA <= 1 && desiredBgLuminanceAA >= 0) {
        var newBgColorAA = axs.utils.translateColor(bgYCC, desiredBgLuminanceAA);
        var newContrastRatioAA = axs.utils.calculateContrastRatio(fgColor, newBgColorAA);
        var suggestedColorsAA = {};
        suggestedColorsAA['bg'] = axs.utils.colorToString(newBgColorAA);
        suggestedColorsAA['fg'] = axs.utils.colorToString(fgColor);
        suggestedColorsAA['contrast'] = newContrastRatioAA.toFixed(2);
        colors['AA'] = suggestedColorsAA;
    }
    if (!('AAA' in colors) && axs.utils.isLowContrast(contrastRatio, style, true) &&
        desiredBgLuminanceAAA <= 1 && desiredBgLuminanceAAA >= 0) {
        var newBgColorAAA = axs.utils.translateColor(bgYCC, desiredBgLuminanceAAA);
        var newContrastRatioAAA = axs.utils.calculateContrastRatio(fgColor, newBgColorAAA);
        var suggestedColorsAAA = {};
        suggestedColorsAAA['bg'] = axs.utils.colorToString(newBgColorAAA);
        suggestedColorsAAA['fg'] = axs.utils.colorToString(fgColor);
        suggestedColorsAAA['contrast'] = newContrastRatioAAA.toFixed(2);
        colors['AAA'] = suggestedColorsAAA;
    }
    return colors;
};

/**
 * Combine the two given color according to alpha blending.
 * @param {axs.utils.Color} fgColor
 * @param {axs.utils.Color} bgColor
 * @return {axs.utils.Color}
 */
axs.utils.flattenColors = function(fgColor, bgColor) {
    var alpha = fgColor.alpha;
    var r = ((1 - alpha) * bgColor.red) + (alpha * fgColor.red);
    var g = ((1 - alpha) * bgColor.green) + (alpha * fgColor.green);
    var b = ((1 - alpha) * bgColor.blue) + (alpha * fgColor.blue);
    var a = fgColor.alpha + (bgColor.alpha * (1 - fgColor.alpha));

    return new axs.utils.Color(r, g, b, a);
};

/**
 * Calculate the luminance of the given color using the WCAG algorithm.
 * @param {axs.utils.Color} color
 * @return {number}
 */
axs.utils.calculateLuminance = function(color) {
/*    var rSRGB = color.red / 255;
    var gSRGB = color.green / 255;
    var bSRGB = color.blue / 255;

    var r = rSRGB <= 0.03928 ? rSRGB / 12.92 : Math.pow(((rSRGB + 0.055)/1.055), 2.4);
    var g = gSRGB <= 0.03928 ? gSRGB / 12.92 : Math.pow(((gSRGB + 0.055)/1.055), 2.4);
    var b = bSRGB <= 0.03928 ? bSRGB / 12.92 : Math.pow(((bSRGB + 0.055)/1.055), 2.4);

    return 0.2126 * r + 0.7152 * g + 0.0722 * b; */
    var ycc = axs.utils.toYCC(color);
    return ycc[0];
};

/**
 * Returns an RGB to YCC conversion matrix for the given kR, kB constants.
 * @param {number} kR
 * @param {number} kB
 * @return {Array.<Array.<number>>}
 */
axs.utils.RGBToYCCMatrix = function(kR, kB) {
    return [
        [
            kR,
            (1 - kR - kB),
            kB
        ],
        [
            -kR/(2 - 2*kB),
            (kR + kB - 1)/(2 - 2*kB),
            (1 - kB)/(2 - 2*kB)
        ],
        [
            (1 - kR)/(2 - 2*kR),
            (kR + kB - 1)/(2 - 2*kR),
            -kB/(2 - 2*kR)
        ]
    ];
};

/**
 * Return the inverse of the given 3x3 matrix.
 * @param {Array.<Array.<number>>} matrix
 * @return Array.<Array.<number>> The inverse of the given matrix.
 */
axs.utils.invert3x3Matrix = function(matrix) {
    var a = matrix[0][0];
    var b = matrix[0][1];
    var c = matrix[0][2];
    var d = matrix[1][0];
    var e = matrix[1][1];
    var f = matrix[1][2];
    var g = matrix[2][0];
    var h = matrix[2][1];
    var k = matrix[2][2];

    var A = (e*k - f*h);
    var B = (f*g - d*k);
    var C = (d*h - e*g);
    var D = (c*h - b*k);
    var E = (a*k - c*g);
    var F = (g*b - a*h);
    var G = (b*f - c*e);
    var H = (c*d - a*f);
    var K = (a*e - b*d);

    var det = a * (e*k - f*h) - b * (k*d - f*g) + c * (d*h - e*g);
    var z = 1/det;

    return axs.utils.scalarMultiplyMatrix([
        [ A, D, G ],
        [ B, E, H ],
        [ C, F, K ]
    ], z);
};

axs.utils.scalarMultiplyMatrix = function(matrix, scalar) {
    var result = [];
    result[0] = [];
    result[1] = [];
    result[2] = [];

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }

    return result;
};

axs.utils.kR = 0.2126;
axs.utils.kB = 0.0722;
axs.utils.YCC_MATRIX = axs.utils.RGBToYCCMatrix(axs.utils.kR, axs.utils.kB);
axs.utils.INVERTED_YCC_MATRIX = axs.utils.invert3x3Matrix(axs.utils.YCC_MATRIX);

/**
 * Multiply the given color vector by the given transformation matrix.
 * @param {Array.<Array.<number>>} matrix A 3x3 conversion matrix
 * @param {Array.<number>} vector A 3-element color vector
 * @return {Array.<number>} A 3-element color vector
 */
axs.utils.convertColor = function(matrix, vector) {
    var a = matrix[0][0];
    var b = matrix[0][1];
    var c = matrix[0][2];
    var d = matrix[1][0];
    var e = matrix[1][1];
    var f = matrix[1][2];
    var g = matrix[2][0];
    var h = matrix[2][1];
    var k = matrix[2][2];

    var x = vector[0];
    var y = vector[1];
    var z = vector[2];

    return [
        a*x + b*y + c*z,
        d*x + e*y + f*z,
        g*x + h*y + k*z
    ];
};

axs.utils.multiplyMatrices = function(matrix1, matrix2) {
    var result = [];
    result[0] = [];
    result[1] = [];
    result[2] = [];

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            result[i][j] = matrix1[i][0] * matrix2[0][j] +
                           matrix1[i][1] * matrix2[1][j] +
                           matrix1[i][2] * matrix2[2][j];
        }
    }
    return result;
};

/**
 * Convert a given RGB color to YCC.
 * @param {axs.utils.Color} color
 */
axs.utils.toYCC = function(color) {
    var rSRGB = color.red / 255;
    var gSRGB = color.green / 255;
    var bSRGB = color.blue / 255;

    var r = rSRGB <= 0.03928 ? rSRGB / 12.92 : Math.pow(((rSRGB + 0.055)/1.055), 2.4);
    var g = gSRGB <= 0.03928 ? gSRGB / 12.92 : Math.pow(((gSRGB + 0.055)/1.055), 2.4);
    var b = bSRGB <= 0.03928 ? bSRGB / 12.92 : Math.pow(((bSRGB + 0.055)/1.055), 2.4);

    return axs.utils.convertColor(axs.utils.YCC_MATRIX, [r, g, b]);
};

/**
 * Convert a color from a YCC color (as a vector) to an RGB color
 * @param {Array.<number>} yccColor
 * @return {axs.utils.Color}
 */
axs.utils.fromYCC = function(yccColor) {
    var rgb = axs.utils.convertColor(axs.utils.INVERTED_YCC_MATRIX, yccColor);

    var r = rgb[0];
    var g = rgb[1];
    var b = rgb[2];
    var rSRGB = r <= 0.00303949 ? (r * 12.92) : (Math.pow(r, (1/2.4)) * 1.055) - 0.055;
    var gSRGB = g <= 0.00303949 ? (g * 12.92) : (Math.pow(g, (1/2.4)) * 1.055) - 0.055;
    var bSRGB = b <= 0.00303949 ? (b * 12.92) : (Math.pow(b, (1/2.4)) * 1.055) - 0.055;

    var red = Math.min(Math.max(Math.round(rSRGB * 255), 0), 255);
    var green = Math.min(Math.max(Math.round(gSRGB * 255), 0), 255);
    var blue = Math.min(Math.max(Math.round(bSRGB * 255), 0), 255);

    return new axs.utils.Color(red, green, blue, 1);
};

axs.utils.scalarMultiplyMatrix = function(matrix, scalar) {
    var result = [];
    result[0] = [];
    result[1] = [];
    result[2] = [];

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }

    return result;
};

axs.utils.multiplyMatrices = function(matrix1, matrix2) {
    var result = [];
    result[0] = [];
    result[1] = [];
    result[2] = [];

    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            result[i][j] = matrix1[i][0] * matrix2[0][j] +
                           matrix1[i][1] * matrix2[1][j] +
                           matrix1[i][2] * matrix2[2][j];
        }
    }
    return result;
};

/**
 * @param {Element} element
 * @return {?number}
 */
axs.utils.getContrastRatioForElement = function(element) {
    var style = window.getComputedStyle(element, null);
    return axs.utils.getContrastRatioForElementWithComputedStyle(style, element);
};

/**
 * @param {CSSStyleDeclaration} style
 * @param {Element} element
 * @return {?number}
 */
axs.utils.getContrastRatioForElementWithComputedStyle = function(style, element) {
    if (axs.utils.isElementHidden(element))
        return null;

    var bgColor = axs.utils.getBgColor(style, element);
    if (!bgColor)
        return null;

    var fgColor = axs.utils.getFgColor(style, element, bgColor);
    if (!fgColor)
        return null;

    return axs.utils.calculateContrastRatio(fgColor, bgColor);
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.isNativeTextElement = function(element) {
    var tagName = element.tagName.toLowerCase();
    var type = element.type ? element.type.toLowerCase() : '';
    if (tagName == 'textarea')
        return true;
    if (tagName != 'input')
        return false;

    switch (type) {
    case 'email':
    case 'number':
    case 'password':
    case 'search':
    case 'text':
    case 'tel':
    case 'url':
    case '':
        return true;
    default:
        return false;
    }
};

/**
 * @param {number} contrastRatio
 * @param {CSSStyleDeclaration} style
 * @param {boolean=} opt_strict Whether to use AA (false) or AAA (true) level
 * @return {boolean}
 */
axs.utils.isLowContrast = function(contrastRatio, style, opt_strict) {
    // Round to nearest 0.1
    var roundedContrastRatio = (Math.round(contrastRatio * 10) / 10);
    if (!opt_strict) {
        return roundedContrastRatio < 3.0 ||
            (!axs.utils.isLargeFont(style) && roundedContrastRatio < 4.5);
    } else {
        return roundedContrastRatio < 4.5 ||
            (!axs.utils.isLargeFont(style) && roundedContrastRatio < 7.0);
    }
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.hasLabel = function(element) {
    var tagName = element.tagName.toLowerCase();
    var type = element.type ? element.type.toLowerCase() : '';

    if (element.hasAttribute('aria-label'))
        return true;
    if (element.hasAttribute('title'))
        return true;
    if (tagName == 'img' && element.hasAttribute('alt'))
        return true;
    if (tagName == 'input' && type == 'image' && element.hasAttribute('alt'))
        return true;
    if (tagName == 'input' && (type == 'submit' || type == 'reset'))
        return true;

    // There's a separate audit that makes sure this points to an actual element or elements.
    if (element.hasAttribute('aria-labelledby'))
        return true;

    if (element.hasAttribute('id')) {
        var labelsFor = document.querySelectorAll('label[for="' + element.id + '"]');
        if (labelsFor.length > 0)
            return true;
    }

    var parent = axs.utils.parentElement(element);
    while (parent) {
        if (parent.tagName.toLowerCase() == 'label') {
            var parentLabel = /** HTMLLabelElement */ parent;
            if (parentLabel.control == element)
                return true;
        }
        parent = axs.utils.parentElement(parent);
    }
    return false;
};

/**
 * @param {Element} element An element to check.
 * @return {boolean} True if the element is hidden from accessibility.
 */
axs.utils.isElementHidden = function(element) {
    if (!(element instanceof element.ownerDocument.defaultView.HTMLElement))
      return false;

    if (element.hasAttribute('chromevoxignoreariahidden'))
        var chromevoxignoreariahidden = true;

    var style = window.getComputedStyle(element, null);
    if (style.display == 'none' || style.visibility == 'hidden')
        return true;

    if (element.hasAttribute('aria-hidden') &&
        element.getAttribute('aria-hidden').toLowerCase() == 'true') {
        return !chromevoxignoreariahidden;
    }

    return false;
};

/**
 * @param {Element} element An element to check.
 * @return {boolean} True if the element or one of its ancestors is
 *     hidden from accessibility.
 */
axs.utils.isElementOrAncestorHidden = function(element) {
    if (axs.utils.isElementHidden(element))
        return true;

    if (axs.utils.parentElement(element))
        return axs.utils.isElementOrAncestorHidden(axs.utils.parentElement(element));
    else
        return false;
};

/**
 * @param {Element} element An element to check
 * @return {boolean} True if the given element is an inline element, false
 *     otherwise.
 */
axs.utils.isInlineElement = function(element) {
    var tagName = element.tagName.toUpperCase();
    return axs.constants.InlineElements[tagName];
};

/**
 * @param {Element} element
 * @return {Object|boolean}
 */
axs.utils.getRoles = function(element) {
    if (!element.hasAttribute('role'))
        return false;
    var roleValue = element.getAttribute('role');
    var roleNames = roleValue.split(' ');
    var roles = [];
    var valid = true;
    for (var i = 0; i < roleNames.length; i++) {
        var role = roleNames[i];
        if (axs.constants.ARIA_ROLES[role])
            roles.push({'name': role, 'details': axs.constants.ARIA_ROLES[role], 'valid': true});
        else {
            roles.push({'name': role, 'valid': false});
            valid = false;
        }
    }

    return { 'roles': roles, 'valid': valid };
};

/**
 * @param {!string} propertyName
 * @param {!string} value
 * @param {!Element} element
 * @return {!Object}
 */
axs.utils.getAriaPropertyValue = function(propertyName, value, element) {
    var propertyKey = propertyName.replace(/^aria-/, '');
    var property = axs.constants.ARIA_PROPERTIES[propertyKey];
    var result = { 'name': propertyName, 'rawValue': value };
    if (!property) {
        result.valid = false;
        result.reason = '"' + propertyName + '" is not a valid ARIA property';
        return result;
    }

    var propertyType = property.valueType;
    if (!propertyType) {
        result.valid = false;
        result.reason = '"' + propertyName + '" is not a valid ARIA property';
        return result;
    }

    switch (propertyType) {
    case "idref":
        var isValid = axs.utils.isValidIDRefValue(value, element);
        result.valid = isValid.valid;
        result.reason = isValid.reason;
        result.idref = isValid.idref;
    case "idref_list":
        var idrefValues = value.split(/\s+/);
        result.valid = true;
        for (var i = 0; i < idrefValues.length; i++) {
            var refIsValid = axs.utils.isValidIDRefValue(idrefValues[i],  element);
            if (!refIsValid.valid)
                result.valid = false;
            if (result.values)
                result.values.push(refIsValid);
            else
                result.values = [refIsValid];
        }
        return result;
    case "integer":
    case "decimal":
        var validNumber = axs.utils.isValidNumber(value);
        if (!validNumber.valid) {
            result.valid = false;
            result.reason = validNumber.reason;
            return result;
        }
        if (Math.floor(validNumber.value) != validNumber.value) {
            result.valid = false;
            result.reason = '' + value + ' is not a whole integer';
        } else {
            result.valid = true;
            result.value = validNumber.value;
        }
        return result;
    case "number":
        var validNumber = axs.utils.isValidNumber(value);
        if (validNumber.valid) {
            result.valid = true;
            result.value = validNumber.value;
        }
    case "string":
        result.valid = true;
        result.value = value;
        return result;
    case "token":
        var validTokenValue = axs.utils.isValidTokenValue(propertyName, value.toLowerCase());
        if (validTokenValue.valid) {
            result.valid = true;
            result.value = validTokenValue.value;
            return result;
        } else {
            result.valid = false;
            result.value = value;
            result.reason = validTokenValue.reason;
            return result;
        }
    case "token_list":
        var tokenValues = value.split(/\s+/);
        result.valid = true;
        for (var i = 0; i < tokenValues.length; i++) {
            var validTokenValue = axs.utils.isValidTokenValue(propertyName, tokenValues[i].toLowerCase());
            if (!validTokenValue.valid) {
                result.valid = false;
                if (result.reason) {
                    result.reason = [ result.reason ];
                    result.reason.push(validTokenValue.reason);
                } else {
                    result.reason = validTokenValue.reason;
                    result.possibleValues = validTokenValue.possibleValues;
                }
            }
            // TODO (more structured result)
            if (result.values)
                result.values.push(validTokenValue.value);
            else
                result.values = [validTokenValue.value];
        }
        return result;
    case "tristate":
        var validTristate = axs.utils.isPossibleValue(value.toLowerCase(), axs.constants.MIXED_VALUES, propertyName);
        if (validTristate.valid) {
            result.valid = true;
            result.value = validTristate.value;
        } else {
            result.valid = false;
            result.value = value;
            result.reason = validTristate.reason;
        }
        return result;
    case "boolean":
        var validBoolean = axs.utils.isValidBoolean(value);
        if (validBoolean.valid) {
            result.valid = true;
            result.value = validBoolean.value;
        } else {
            result.valid = false;
            result.value = value;
            result.reason = validBoolean.reason;
        }
        return result;
    }
    result.valid = false;
    result.reason = 'Not a valid ARIA property';
    return result;
};

/**
 * @param {string} propertyName The name of the property.
 * @param {string} value The value to check.
 * @return {!Object}
 */
axs.utils.isValidTokenValue = function(propertyName, value) {
    var propertyKey = propertyName.replace(/^aria-/, '');
    var propertyDetails = axs.constants.ARIA_PROPERTIES[propertyKey];
    var possibleValues = propertyDetails.valuesSet;
    return axs.utils.isPossibleValue(value, possibleValues, propertyName);
};

/**
 * @param {string} value
 * @param {Object.<string, boolean>} possibleValues
 * @param {string} propertyName The name of the property.
 * @return {!Object}
 */
axs.utils.isPossibleValue = function(value, possibleValues, propertyName) {
    if (!possibleValues[value])
        return { 'valid': false,
                 'value': value,
                 'reason': '"' + value + '" is not a valid value for ' + propertyName,
                 'possibleValues': Object.keys(possibleValues) };
    return { 'valid': true, 'value': value };
};

/**
 * @param {string} value
 * @return {!Object}
 */
axs.utils.isValidBoolean = function(value) {
    try {
        var parsedValue = JSON.parse(value);
    } catch (e) {
        parsedValue = '';
    }
    if (typeof(parsedValue) != 'boolean')
        return { 'valid': false,
                 'value': value,
                 'reason': '"' + value + '" is not a true/false value' };
    return { 'valid': true, 'value': parsedValue };
};

/**
 * @param {string} value
 * @param {!Element} element
 * @return {!Object}
 */
axs.utils.isValidIDRefValue = function(value, element) {
    if (value.length == 0)
        return { 'valid': true, 'idref': value };
    if (!element.ownerDocument.getElementById(value))
        return { 'valid': false,
                 'idref': value,
                 'reason': 'No element with ID "' + value + '"' };
    return { 'valid': true, 'idref': value };
};

/**
 * @param {string} value
 * @return {!Object}
 */
axs.utils.isValidNumber = function(value) {
    try {
        var parsedValue = JSON.parse(value);
    } catch (ex) {
        return { 'valid': false,
                 'value': value,
                 'reason': '"' + value + '" is not a number' };
    }
    if (typeof(parsedValue) != 'number') {
        return { 'valid': false,
                 'value': value,
                 'reason': '"' + value + '" is not a number' };
    }
    return { 'valid': true, 'value': parsedValue };
};

/**
 * @param {Element} element
 * @return {boolean}
 */
axs.utils.isElementImplicitlyFocusable = function(element) {
    var defaultView = element.ownerDocument.defaultView;

    if (element instanceof defaultView.HTMLAnchorElement ||
        element instanceof defaultView.HTMLAreaElement)
        return element.hasAttribute('href');
    if (element instanceof defaultView.HTMLInputElement ||
        element instanceof defaultView.HTMLSelectElement ||
        element instanceof defaultView.HTMLTextAreaElement ||
        element instanceof defaultView.HTMLButtonElement ||
        element instanceof defaultView.HTMLIFrameElement)
        return !element.disabled;
    return false;
};

/**
 * Returns an array containing the values of the given JSON-compatible object.
 * (Simply ignores any function values.)
 * @param {Object} obj
 * @return {Array}
 */
axs.utils.values = function(obj) {
    var values = [];
    for (var key in obj) {
        if (obj.hasOwnProperty(key) && typeof obj[key] != 'function')
            values.push(obj[key]);
    }
    return values;
};

/**
 * Returns an object containing the same keys and values as the given
 * JSON-compatible object. (Simply ignores any function values.)
 * @param {Object} obj
 * @return {Object}
 */
axs.utils.namedValues = function(obj) {
    var values = {};
    for (var key in obj) {
        if (obj.hasOwnProperty(key) && typeof obj[key] != 'function')
            values[key] = obj[key];
    }
    return values;
};

/** Gets a CSS selector text for a DOM object.
 * @param {Node} obj The DOM object.
 * @return {string} CSS selector text for the DOM object.
 */
axs.utils.getQuerySelectorText = function(obj) {
  if (obj == null || obj.tagName == 'HTML') {
    return 'html';
  } else if (obj.tagName == 'BODY') {
    return 'body';
  }

  if (obj.hasAttribute) {
    if (obj.id) {
      return '#' + obj.id;
    }

    if (obj.className) {
      var selector = '';
      for (var i = 0; i < obj.classList.length; i++)
        selector += '.' + obj.classList[i];

      var total = 0;
      if (obj.parentNode) {
        for (i = 0; i < obj.parentNode.children.length; i++) {
          var similar = obj.parentNode.children[i];
          if (axs.browserUtils.matchSelector(similar, selector))
            total++;
          if (similar === obj)
            break;
        }
      } else {
        total = 1;
      }

      if (total == 1) {
        return axs.utils.getQuerySelectorText(obj.parentNode) +
               ' > ' + selector;
      }
    }

    if (obj.parentNode) {
      var similarTags = obj.parentNode.children;
      var total = 1;
      var i = 0;
      while (similarTags[i] !== obj) {
        if (similarTags[i].tagName == obj.tagName) {
          total++;
        }
        i++;
      }

      var next = '';
      if (obj.parentNode.tagName != 'BODY') {
        next = axs.utils.getQuerySelectorText(obj.parentNode) +
               ' > ';
      }

      if (total == 1) {
        return next +
               obj.tagName;
      } else {
        return next +
               obj.tagName +
               ':nth-of-type(' + total + ')';
      }
    }

  } else if (obj.selectorText) {
    return obj.selectorText;
  }

  return '';
};

/**
 * Gets elements that refer to this element in an ARIA attribute that takes an
 * ID reference list or single ID reference.
 * @param {!string} attributeName Name of an ARIA attribute, e.g. 'aria-owns'.
 * @param {!Element} element a potential referent.
 * @return {NodeList} The elements that refer to this element.
 */
axs.utils.getIdReferrers = function(attributeName, element) {
    var id = element.id;
    var propertyKey = attributeName.replace(/^aria-/, '');
    var property = axs.constants.ARIA_PROPERTIES[propertyKey];
    if (!id || !property)
        return null;
    var propertyType = property.valueType;
    if (propertyType === 'idref_list' || propertyType === 'idref') {
        id = id.replace(/'/g, "\\'");
        var referrerQuery = "[" + attributeName + "~='" + id + "']";
        return element.ownerDocument.querySelectorAll(referrerQuery);
    }
    return null;
};

/**
 * Gets a subset of 'axs.constants.ARIA_PROPERTIES' filtered by 'valueType'.
 * @param {!Array.<string>} valueTypes Types to match, e.g. ['idref_list'].
 * @return {Object.<string, Object>} axs.constants.ARIA_PROPERTIES which match.
 */
axs.utils.getAriaPropertiesByValueType = function(valueTypes) {
    var result = {};
    for (var propertyName in axs.constants.ARIA_PROPERTIES) {
        var property = axs.constants.ARIA_PROPERTIES[propertyName];
        if (property && valueTypes.indexOf(property.valueType) >= 0) {
            result[propertyName] = property;
        }
    }
    return result;
};

/**
 * Builds a selector that matches an element with any of these ARIA properties.
 * @param {Object.<string, Object>} ariaProperties axs.constants.ARIA_PROPERTIES
 * @return {!string} The selector.
 */
axs.utils.getSelectorForAriaProperties = function(ariaProperties) {
    var propertyNames = Object.keys(/** @type {!Object} */(ariaProperties));
    var result = propertyNames.map(function(propertyName) {
        return '[aria-' + propertyName + ']';
    });
    result.sort();  // facilitates reading long selectors and unit testing
    return result.join(',');
};
