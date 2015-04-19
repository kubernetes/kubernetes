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
goog.require('axs.properties');
goog.require('axs.utils');

/**
 * @type {axs.AuditRule.Spec}
 */
axs.AuditRule.specs.lowContrastElements = {
    name: 'lowContrastElements',
    heading: 'Text elements should have a reasonable contrast ratio',
    url: 'https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_color_01--text-elements-should-have-a-reasonable-contrast-ratio',
    severity: axs.constants.Severity.WARNING,
    relevantElementMatcher: function(element) {
        return axs.properties.hasDirectTextDescendant(element);
    },
    test: function(element) {
        var style = window.getComputedStyle(element, null);
        var contrastRatio =
            axs.utils.getContrastRatioForElementWithComputedStyle(style, element);
        return (contrastRatio && axs.utils.isLowContrast(contrastRatio, style));
    },
    code: 'AX_COLOR_01'
};
