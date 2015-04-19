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

goog.require('axs.AuditRule');
goog.require('axs.AuditRules');
goog.require('axs.browserUtils');
goog.require('axs.constants.Severity');
goog.require('axs.utils');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.focusableElementNotVisibleAndNotAriaHidden = {
    name: 'focusableElementNotVisibleAndNotAriaHidden',
    heading: 'These elements are focusable but either invisible or obscured by another element',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_focus_01--these-elements-are-focusable-but-either-invisible-or-obscured-by-another-element',
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        var isFocusable = axs.browserUtils.matchSelector(
            element, axs.utils.FOCUSABLE_ELEMENTS_SELECTOR);
        if (!isFocusable)
            return false;
        if (element.tabIndex >= 0)
            return true;
        // Ignore elements which have negative tabindex and an ancestor with a
        // widget role, since they can be accessed neither with tab nor with
        // a screen reader
        for (var parent = axs.utils.parentElement(element); parent != null;
             parent = axs.utils.parentElement(parent)) {
            if (axs.utils.elementIsAriaWidget(parent))
                return false;
        }
        // Ignore elements which have a negative tabindex and no text content,
        // as they will be skipped by assistive technology
        if (axs.properties.findTextAlternatives(element, {}).trim() === '')
            return false;

        return true;

    },
    test: function(element) {
        if (axs.utils.isElementOrAncestorHidden(element))
            return false;
        element.focus();
        return !axs.utils.elementIsVisible(element)
    },
    code: 'AX_FOCUS_01'
};
