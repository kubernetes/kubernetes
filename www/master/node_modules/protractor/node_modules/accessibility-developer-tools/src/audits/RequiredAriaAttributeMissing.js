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
goog.require('axs.constants');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.requiredAriaAttributeMissing = {
    name: 'requiredAriaAttributeMissing',
    heading: 'Elements with ARIA roles must have all required attributes for that role',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_aria_03--elements-with-aria-roles-must-have-all-required-attributes-for-that-role',
    severity: axs.constants.Severity.SEVERE,
    relevantElementMatcher: function(element) {
        return axs.browserUtils.matchSelector(element, '[role]');
    },
    test: function(element) {
        var roles = axs.utils.getRoles(element);
        if (!roles.valid)
            return false;  // That's a different error.
        for (var i = 0; i < roles.roles.length; i++) {
            var role = roles.roles[i];
            var requiredProperties = role.details.requiredPropertiesSet;
            for (var property in requiredProperties) {
                var propertyKey = property.replace(/^aria-/, '');
                var propertyDetails = axs.constants.ARIA_PROPERTIES[propertyKey];
                if ('defaultValue' in propertyDetails)
                    continue;
                if (!element.hasAttribute(property))
                    return true;
            }
        }
    },
    code: 'AX_ARIA_03'
};
