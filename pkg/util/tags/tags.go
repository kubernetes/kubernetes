package tags

import "strings"

// GetAdditionalTagsFromAnnotation converts the comma separated list of key-value
// pairs in the "AdditionalTags" annotation and returns it as a map.
func GetAdditionalTagsFromAnnotation(annotations map[string]string, annotationName string) map[string]string {
	additionalTags := make(map[string]string)
	if additionalTagsList, ok := annotations[annotationName]; ok {
		additionalTagsList = strings.TrimSpace(additionalTagsList)

		// Break up list of "Key1=Val,Key2=Val2"
		tagList := strings.Split(additionalTagsList, ",")

		// Break up "Key=Val"
		for _, tagSet := range tagList {
			tag := strings.Split(strings.TrimSpace(tagSet), "=")

			// Accept "Key=val" or "Key=" or just "Key"
			if len(tag) >= 2 && len(tag[0]) != 0 {
				// There is a key and a value, so save it
				additionalTags[tag[0]] = tag[1]
			} else if len(tag) == 1 && len(tag[0]) != 0 {
				// Just "Key"
				additionalTags[tag[0]] = ""
			}
		}
	}

	return additionalTags
}
