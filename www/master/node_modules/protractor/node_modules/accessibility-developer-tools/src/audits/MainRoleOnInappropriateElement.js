// Copyright 2013 Google Inc.
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
goog.require('axs.properties');
goog.require('axs.utils');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.mainRoleOnInappropriateElement = {
    name: 'mainRoleOnInappropriateElement',
    heading: 'role=main should only appear on significant elements',
    url: '',
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        return axs.browserUtils.matchSelector(element, '[role~=main]');
    },
    test: function(element) {
        if (axs.utils.isInlineElement(element))
            return true;
        var computedTextContent = axs.properties.getTextFromDescendantContent(element);
        if (!computedTextContent || computedTextContent.length < 50)
            return true;

        return false;
    },
    code: 'AX_ARIA_04'
};
