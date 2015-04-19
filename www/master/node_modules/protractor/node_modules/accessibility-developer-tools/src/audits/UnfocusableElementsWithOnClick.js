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
goog.require('axs.constants.Severity');
goog.require('axs.utils');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.unfocusableElementsWithOnClick = {
    name: 'unfocusableElementsWithOnClick',
    heading: 'Elements with onclick handlers must be focusable',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_focus_02--elements-with-onclick-handlers-must-be-focusable',
    severity: axs.constants.Severity.WARNING,
    opt_requiresConsoleAPI: true,
    relevantElementMatcher: function(element) {
        // element.ownerDocument may not be current document if it is in an iframe
        if (element instanceof element.ownerDocument.defaultView.HTMLBodyElement) {
            return false;
        }
        if (axs.utils.isElementOrAncestorHidden(element)) {
            return false;
        }
        var eventListeners = getEventListeners(element);
        if ('click' in eventListeners) {
            return true;
        }
        return false;
    },
    test: function(element) {
        return !element.hasAttribute('tabindex') &&
               !axs.utils.isElementImplicitlyFocusable(element) &&
               !element.disabled;
    },
    code: 'AX_FOCUS_02'
};
