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
axs.AuditRule.specs.controlsWithoutLabel = {
    name: 'controlsWithoutLabel',
    heading: 'Controls and media elements should have labels',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_text_01--controls-and-media-elements-should-have-labels',
    severity: axs.constants.Severity.SEVERE,
    relevantElementMatcher: function(element) {
        var controlsSelector = ['input:not([type="hidden"]):not([disabled])',
                                'select:not([disabled])',
                                'textarea:not([disabled])',
                                'button:not([disabled])',
                                'video:not([disabled])'].join(', ');
        var isControl = axs.browserUtils.matchSelector(element, controlsSelector);
        if (!isControl || element.getAttribute('role') == 'presentation')
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
        return true;
    },
    test: function(control) {
        if (axs.utils.isElementOrAncestorHidden(control))
            return false;
        if (control.tagName.toLowerCase() == 'input' &&
            control.type == 'button' &&
            control.value.length) {
            return false;
        }
        if (control.tagName.toLowerCase() == 'button') {
            var innerText = control.textContent.replace(/^\s+|\s+$/g, '');
            if (innerText.length)
                return false;
        }
        if (!axs.utils.hasLabel(control))
            return true;
        return false;
    },
    code: 'AX_TEXT_01',
    ruleName: 'Controls and media elements should have labels'
};
