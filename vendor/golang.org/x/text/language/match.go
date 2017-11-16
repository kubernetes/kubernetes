// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import "errors"

// A MatchOption configures a Matcher.
type MatchOption func(*matcher)

// PreferSameScript will, in the absence of a match, result in the first
// preferred tag with the same script as a supported tag to match this supported
// tag. The default is currently true, but this may change in the future.
func PreferSameScript(preferSame bool) MatchOption {
	return func(m *matcher) { m.preferSameScript = preferSame }
}

// Matcher is the interface that wraps the Match method.
//
// Match returns the best match for any of the given tags, along with
// a unique index associated with the returned tag and a confidence
// score.
type Matcher interface {
	Match(t ...Tag) (tag Tag, index int, c Confidence)
}

// Comprehends reports the confidence score for a speaker of a given language
// to being able to comprehend the written form of an alternative language.
func Comprehends(speaker, alternative Tag) Confidence {
	_, _, c := NewMatcher([]Tag{alternative}).Match(speaker)
	return c
}

// NewMatcher returns a Matcher that matches an ordered list of preferred tags
// against a list of supported tags based on written intelligibility, closeness
// of dialect, equivalence of subtags and various other rules. It is initialized
// with the list of supported tags. The first element is used as the default
// value in case no match is found.
//
// Its Match method matches the first of the given Tags to reach a certain
// confidence threshold. The tags passed to Match should therefore be specified
// in order of preference. Extensions are ignored for matching.
//
// The index returned by the Match method corresponds to the index of the
// matched tag in t, but is augmented with the Unicode extension ('u')of the
// corresponding preferred tag. This allows user locale options to be passed
// transparently.
func NewMatcher(t []Tag, options ...MatchOption) Matcher {
	return newMatcher(t, options)
}

func (m *matcher) Match(want ...Tag) (t Tag, index int, c Confidence) {
	match, w, c := m.getBest(want...)
	if match != nil {
		t, index = match.tag, match.index
	} else {
		// TODO: this should be an option
		t = m.default_.tag
		if m.preferSameScript {
		outer:
			for _, w := range want {
				script, _ := w.Script()
				if script.scriptID == 0 {
					// Don't do anything if there is no script, such as with
					// private subtags.
					continue
				}
				for i, h := range m.supported {
					if script.scriptID == h.maxScript {
						t, index = h.tag, i
						break outer
					}
				}
			}
		}
		// TODO: select first language tag based on script.
	}
	if w.region != 0 && t.region != 0 && t.region.contains(w.region) {
		t, _ = Raw.Compose(t, Region{w.region})
	}
	// Copy options from the user-provided tag into the result tag. This is hard
	// to do after the fact, so we do it here.
	// TODO: add in alternative variants to -u-va-.
	// TODO: add preferred region to -u-rg-.
	// TODO: add other extensions. Merge with existing extensions.
	if u, ok := w.Extension('u'); ok {
		t, _ = Raw.Compose(t, u)
	}
	return t, index, c
}

type scriptRegionFlags uint8

const (
	isList = 1 << iota
	scriptInFrom
	regionInFrom
)

func (t *Tag) setUndefinedLang(id langID) {
	if t.lang == 0 {
		t.lang = id
	}
}

func (t *Tag) setUndefinedScript(id scriptID) {
	if t.script == 0 {
		t.script = id
	}
}

func (t *Tag) setUndefinedRegion(id regionID) {
	if t.region == 0 || t.region.contains(id) {
		t.region = id
	}
}

// ErrMissingLikelyTagsData indicates no information was available
// to compute likely values of missing tags.
var ErrMissingLikelyTagsData = errors.New("missing likely tags data")

// addLikelySubtags sets subtags to their most likely value, given the locale.
// In most cases this means setting fields for unknown values, but in some
// cases it may alter a value.  It returns a ErrMissingLikelyTagsData error
// if the given locale cannot be expanded.
func (t Tag) addLikelySubtags() (Tag, error) {
	id, err := addTags(t)
	if err != nil {
		return t, err
	} else if id.equalTags(t) {
		return t, nil
	}
	id.remakeString()
	return id, nil
}

// specializeRegion attempts to specialize a group region.
func specializeRegion(t *Tag) bool {
	if i := regionInclusion[t.region]; i < nRegionGroups {
		x := likelyRegionGroup[i]
		if langID(x.lang) == t.lang && scriptID(x.script) == t.script {
			t.region = regionID(x.region)
		}
		return true
	}
	return false
}

func addTags(t Tag) (Tag, error) {
	// We leave private use identifiers alone.
	if t.private() {
		return t, nil
	}
	if t.script != 0 && t.region != 0 {
		if t.lang != 0 {
			// already fully specified
			specializeRegion(&t)
			return t, nil
		}
		// Search matches for und-script-region. Note that for these cases
		// region will never be a group so there is no need to check for this.
		list := likelyRegion[t.region : t.region+1]
		if x := list[0]; x.flags&isList != 0 {
			list = likelyRegionList[x.lang : x.lang+uint16(x.script)]
		}
		for _, x := range list {
			// Deviating from the spec. See match_test.go for details.
			if scriptID(x.script) == t.script {
				t.setUndefinedLang(langID(x.lang))
				return t, nil
			}
		}
	}
	if t.lang != 0 {
		// Search matches for lang-script and lang-region, where lang != und.
		if t.lang < langNoIndexOffset {
			x := likelyLang[t.lang]
			if x.flags&isList != 0 {
				list := likelyLangList[x.region : x.region+uint16(x.script)]
				if t.script != 0 {
					for _, x := range list {
						if scriptID(x.script) == t.script && x.flags&scriptInFrom != 0 {
							t.setUndefinedRegion(regionID(x.region))
							return t, nil
						}
					}
				} else if t.region != 0 {
					count := 0
					goodScript := true
					tt := t
					for _, x := range list {
						// We visit all entries for which the script was not
						// defined, including the ones where the region was not
						// defined. This allows for proper disambiguation within
						// regions.
						if x.flags&scriptInFrom == 0 && t.region.contains(regionID(x.region)) {
							tt.region = regionID(x.region)
							tt.setUndefinedScript(scriptID(x.script))
							goodScript = goodScript && tt.script == scriptID(x.script)
							count++
						}
					}
					if count == 1 {
						return tt, nil
					}
					// Even if we fail to find a unique Region, we might have
					// an unambiguous script.
					if goodScript {
						t.script = tt.script
					}
				}
			}
		}
	} else {
		// Search matches for und-script.
		if t.script != 0 {
			x := likelyScript[t.script]
			if x.region != 0 {
				t.setUndefinedRegion(regionID(x.region))
				t.setUndefinedLang(langID(x.lang))
				return t, nil
			}
		}
		// Search matches for und-region. If und-script-region exists, it would
		// have been found earlier.
		if t.region != 0 {
			if i := regionInclusion[t.region]; i < nRegionGroups {
				x := likelyRegionGroup[i]
				if x.region != 0 {
					t.setUndefinedLang(langID(x.lang))
					t.setUndefinedScript(scriptID(x.script))
					t.region = regionID(x.region)
				}
			} else {
				x := likelyRegion[t.region]
				if x.flags&isList != 0 {
					x = likelyRegionList[x.lang]
				}
				if x.script != 0 && x.flags != scriptInFrom {
					t.setUndefinedLang(langID(x.lang))
					t.setUndefinedScript(scriptID(x.script))
					return t, nil
				}
			}
		}
	}

	// Search matches for lang.
	if t.lang < langNoIndexOffset {
		x := likelyLang[t.lang]
		if x.flags&isList != 0 {
			x = likelyLangList[x.region]
		}
		if x.region != 0 {
			t.setUndefinedScript(scriptID(x.script))
			t.setUndefinedRegion(regionID(x.region))
		}
		specializeRegion(&t)
		if t.lang == 0 {
			t.lang = _en // default language
		}
		return t, nil
	}
	return t, ErrMissingLikelyTagsData
}

func (t *Tag) setTagsFrom(id Tag) {
	t.lang = id.lang
	t.script = id.script
	t.region = id.region
}

// minimize removes the region or script subtags from t such that
// t.addLikelySubtags() == t.minimize().addLikelySubtags().
func (t Tag) minimize() (Tag, error) {
	t, err := minimizeTags(t)
	if err != nil {
		return t, err
	}
	t.remakeString()
	return t, nil
}

// minimizeTags mimics the behavior of the ICU 51 C implementation.
func minimizeTags(t Tag) (Tag, error) {
	if t.equalTags(und) {
		return t, nil
	}
	max, err := addTags(t)
	if err != nil {
		return t, err
	}
	for _, id := range [...]Tag{
		{lang: t.lang},
		{lang: t.lang, region: t.region},
		{lang: t.lang, script: t.script},
	} {
		if x, err := addTags(id); err == nil && max.equalTags(x) {
			t.setTagsFrom(id)
			break
		}
	}
	return t, nil
}

// Tag Matching
// CLDR defines an algorithm for finding the best match between two sets of language
// tags. The basic algorithm defines how to score a possible match and then find
// the match with the best score
// (see http://www.unicode.org/reports/tr35/#LanguageMatching).
// Using scoring has several disadvantages. The scoring obfuscates the importance of
// the various factors considered, making the algorithm harder to understand. Using
// scoring also requires the full score to be computed for each pair of tags.
//
// We will use a different algorithm which aims to have the following properties:
// - clarity on the precedence of the various selection factors, and
// - improved performance by allowing early termination of a comparison.
//
// Matching algorithm (overview)
// Input:
//   - supported: a set of supported tags
//   - default:   the default tag to return in case there is no match
//   - desired:   list of desired tags, ordered by preference, starting with
//                the most-preferred.
//
// Algorithm:
//   1) Set the best match to the lowest confidence level
//   2) For each tag in "desired":
//     a) For each tag in "supported":
//        1) compute the match between the two tags.
//        2) if the match is better than the previous best match, replace it
//           with the new match. (see next section)
//     b) if the current best match is above a certain threshold, return this
//        match without proceeding to the next tag in "desired". [See Note 1]
//   3) If the best match so far is below a certain threshold, return "default".
//
// Ranking:
// We use two phases to determine whether one pair of tags are a better match
// than another pair of tags. First, we determine a rough confidence level. If the
// levels are different, the one with the highest confidence wins.
// Second, if the rough confidence levels are identical, we use a set of tie-breaker
// rules.
//
// The confidence level of matching a pair of tags is determined by finding the
// lowest confidence level of any matches of the corresponding subtags (the
// result is deemed as good as its weakest link).
// We define the following levels:
//   Exact    - An exact match of a subtag, before adding likely subtags.
//   MaxExact - An exact match of a subtag, after adding likely subtags.
//              [See Note 2].
//   High     - High level of mutual intelligibility between different subtag
//              variants.
//   Low      - Low level of mutual intelligibility between different subtag
//              variants.
//   No       - No mutual intelligibility.
//
// The following levels can occur for each type of subtag:
//   Base:    Exact, MaxExact, High, Low, No
//   Script:  Exact, MaxExact [see Note 3], Low, No
//   Region:  Exact, MaxExact, High
//   Variant: Exact, High
//   Private: Exact, No
//
// Any result with a confidence level of Low or higher is deemed a possible match.
// Once a desired tag matches any of the supported tags with a level of MaxExact
// or higher, the next desired tag is not considered (see Step 2.b).
// Note that CLDR provides languageMatching data that defines close equivalence
// classes for base languages, scripts and regions.
//
// Tie-breaking
// If we get the same confidence level for two matches, we apply a sequence of
// tie-breaking rules. The first that succeeds defines the result. The rules are
// applied in the following order.
//   1) Original language was defined and was identical.
//   2) Original region was defined and was identical.
//   3) Distance between two maximized regions was the smallest.
//   4) Original script was defined and was identical.
//   5) Distance from want tag to have tag using the parent relation [see Note 5.]
// If there is still no winner after these rules are applied, the first match
// found wins.
//
// Notes:
// [1] Note that even if we may not have a perfect match, if a match is above a
//     certain threshold, it is considered a better match than any other match
//     to a tag later in the list of preferred language tags.
// [2] In practice, as matching of Exact is done in a separate phase from
//     matching the other levels, we reuse the Exact level to mean MaxExact in
//     the second phase. As a consequence, we only need the levels defined by
//     the Confidence type. The MaxExact confidence level is mapped to High in
//     the public API.
// [3] We do not differentiate between maximized script values that were derived
//     from suppressScript versus most likely tag data. We determined that in
//     ranking the two, one ranks just after the other. Moreover, the two cannot
//     occur concurrently. As a consequence, they are identical for practical
//     purposes.
// [4] In case of deprecated, macro-equivalents and legacy mappings, we assign
//     the MaxExact level to allow iw vs he to still be a closer match than
//     en-AU vs en-US, for example.
// [5] In CLDR a locale inherits fields that are unspecified for this locale
//     from its parent. Therefore, if a locale is a parent of another locale,
//     it is a strong measure for closeness, especially when no other tie
//     breaker rule applies. One could also argue it is inconsistent, for
//     example, when pt-AO matches pt (which CLDR equates with pt-BR), even
//     though its parent is pt-PT according to the inheritance rules.
//
// Implementation Details:
// There are several performance considerations worth pointing out. Most notably,
// we preprocess as much as possible (within reason) at the time of creation of a
// matcher. This includes:
//   - creating a per-language map, which includes data for the raw base language
//     and its canonicalized variant (if applicable),
//   - expanding entries for the equivalence classes defined in CLDR's
//     languageMatch data.
// The per-language map ensures that typically only a very small number of tags
// need to be considered. The pre-expansion of canonicalized subtags and
// equivalence classes reduces the amount of map lookups that need to be done at
// runtime.

// matcher keeps a set of supported language tags, indexed by language.
type matcher struct {
	default_         *haveTag
	supported        []*haveTag
	index            map[langID]*matchHeader
	passSettings     bool
	preferSameScript bool
}

// matchHeader has the lists of tags for exact matches and matches based on
// maximized and canonicalized tags for a given language.
type matchHeader struct {
	exact []*haveTag
	max   []*haveTag
}

// haveTag holds a supported Tag and its maximized script and region. The maximized
// or canonicalized language is not stored as it is not needed during matching.
type haveTag struct {
	tag Tag

	// index of this tag in the original list of supported tags.
	index int

	// conf is the maximum confidence that can result from matching this haveTag.
	// When conf < Exact this means it was inserted after applying a CLDR equivalence rule.
	conf Confidence

	// Maximized region and script.
	maxRegion regionID
	maxScript scriptID

	// altScript may be checked as an alternative match to maxScript. If altScript
	// matches, the confidence level for this match is Low. Theoretically there
	// could be multiple alternative scripts. This does not occur in practice.
	altScript scriptID

	// nextMax is the index of the next haveTag with the same maximized tags.
	nextMax uint16
}

func makeHaveTag(tag Tag, index int) (haveTag, langID) {
	max := tag
	if tag.lang != 0 {
		max, _ = max.canonicalize(All)
		max, _ = addTags(max)
		max.remakeString()
	}
	return haveTag{tag, index, Exact, max.region, max.script, altScript(max.lang, max.script), 0}, max.lang
}

// altScript returns an alternative script that may match the given script with
// a low confidence.  At the moment, the langMatch data allows for at most one
// script to map to another and we rely on this to keep the code simple.
func altScript(l langID, s scriptID) scriptID {
	for _, alt := range matchScript {
		// TODO: also match cases where language is not the same.
		if (langID(alt.wantLang) == l || langID(alt.haveLang) == l) &&
			scriptID(alt.haveScript) == s {
			return scriptID(alt.wantScript)
		}
	}
	return 0
}

// addIfNew adds a haveTag to the list of tags only if it is a unique tag.
// Tags that have the same maximized values are linked by index.
func (h *matchHeader) addIfNew(n haveTag, exact bool) {
	// Don't add new exact matches.
	for _, v := range h.exact {
		if v.tag.equalsRest(n.tag) {
			return
		}
	}
	if exact {
		h.exact = append(h.exact, &n)
	}
	// Allow duplicate maximized tags, but create a linked list to allow quickly
	// comparing the equivalents and bail out.
	for i, v := range h.max {
		if v.maxScript == n.maxScript &&
			v.maxRegion == n.maxRegion &&
			v.tag.variantOrPrivateTagStr() == n.tag.variantOrPrivateTagStr() {
			for h.max[i].nextMax != 0 {
				i = int(h.max[i].nextMax)
			}
			h.max[i].nextMax = uint16(len(h.max))
			break
		}
	}
	h.max = append(h.max, &n)
}

// header returns the matchHeader for the given language. It creates one if
// it doesn't already exist.
func (m *matcher) header(l langID) *matchHeader {
	if h := m.index[l]; h != nil {
		return h
	}
	h := &matchHeader{}
	m.index[l] = h
	return h
}

func toConf(d uint8) Confidence {
	if d <= 10 {
		return High
	}
	if d < 30 {
		return Low
	}
	return No
}

// newMatcher builds an index for the given supported tags and returns it as
// a matcher. It also expands the index by considering various equivalence classes
// for a given tag.
func newMatcher(supported []Tag, options []MatchOption) *matcher {
	m := &matcher{
		index:            make(map[langID]*matchHeader),
		preferSameScript: true,
	}
	for _, o := range options {
		o(m)
	}
	if len(supported) == 0 {
		m.default_ = &haveTag{}
		return m
	}
	// Add supported languages to the index. Add exact matches first to give
	// them precedence.
	for i, tag := range supported {
		pair, _ := makeHaveTag(tag, i)
		m.header(tag.lang).addIfNew(pair, true)
		m.supported = append(m.supported, &pair)
	}
	m.default_ = m.header(supported[0].lang).exact[0]
	for i, tag := range supported {
		pair, max := makeHaveTag(tag, i)
		if max != tag.lang {
			m.header(max).addIfNew(pair, false)
		}
	}

	// TODO: include alt script.
	// - don't replace regions, but allow regions to be made more specific.

	// update is used to add indexes in the map for equivalent languages.
	// If force is true, the update will also apply to derived entries. To
	// avoid applying a "transitive closure", use false.
	update := func(want, have uint16, conf Confidence, force bool) {
		if hh := m.index[langID(have)]; hh != nil {
			if !force && len(hh.exact) == 0 {
				return
			}
			hw := m.header(langID(want))
			for _, ht := range hh.max {
				v := *ht
				if conf < v.conf {
					v.conf = conf
				}
				v.nextMax = 0 // this value needs to be recomputed
				if v.altScript != 0 {
					v.altScript = altScript(langID(want), v.maxScript)
				}
				hw.addIfNew(v, conf == Exact && len(hh.exact) > 0)
			}
		}
	}

	// Add entries for languages with mutual intelligibility as defined by CLDR's
	// languageMatch data.
	for _, ml := range matchLang {
		update(ml.want, ml.have, toConf(ml.distance), false)
		if !ml.oneway {
			update(ml.have, ml.want, toConf(ml.distance), false)
		}
	}

	// Add entries for possible canonicalizations. This is an optimization to
	// ensure that only one map lookup needs to be done at runtime per desired tag.
	// First we match deprecated equivalents. If they are perfect equivalents
	// (their canonicalization simply substitutes a different language code, but
	// nothing else), the match confidence is Exact, otherwise it is High.
	for i, lm := range langAliasMap {
		if lm.from == _sh {
			continue
		}

		// If deprecated codes match and there is no fiddling with the script or
		// or region, we consider it an exact match.
		conf := Exact
		if langAliasTypes[i] != langMacro {
			if !isExactEquivalent(langID(lm.from)) {
				conf = High
			}
			update(lm.to, lm.from, conf, true)
		}
		update(lm.from, lm.to, conf, true)
	}
	return m
}

// getBest gets the best matching tag in m for any of the given tags, taking into
// account the order of preference of the given tags.
func (m *matcher) getBest(want ...Tag) (got *haveTag, orig Tag, c Confidence) {
	best := bestMatch{}
	for _, w := range want {
		var max Tag
		// Check for exact match first.
		h := m.index[w.lang]
		if w.lang != 0 {
			// Base language is defined.
			if h == nil {
				continue
			}
			for i := range h.exact {
				have := h.exact[i]
				if have.tag.equalsRest(w) {
					return have, w, Exact
				}
			}
			max, _ = w.canonicalize(Legacy | Deprecated)
			max, _ = addTags(max)
		} else {
			// Base language is not defined.
			if h != nil {
				for i := range h.exact {
					have := h.exact[i]
					if have.tag.equalsRest(w) {
						return have, w, Exact
					}
				}
			}
			if w.script == 0 && w.region == 0 {
				// We skip all tags matching und for approximate matching, including
				// private tags.
				continue
			}
			max, _ = addTags(w)
			if h = m.index[max.lang]; h == nil {
				continue
			}
		}
		// Check for match based on maximized tag.
		for i := range h.max {
			have := h.max[i]
			best.update(have, w, max.script, max.region)
			if best.conf == Exact {
				for have.nextMax != 0 {
					have = h.max[have.nextMax]
					best.update(have, w, max.script, max.region)
				}
				return best.have, best.want, High
			}
		}
	}
	if best.conf <= No {
		if len(want) != 0 {
			return nil, want[0], No
		}
		return nil, Tag{}, No
	}
	return best.have, best.want, best.conf
}

// bestMatch accumulates the best match so far.
type bestMatch struct {
	have *haveTag
	want Tag
	conf Confidence
	// Cached results from applying tie-breaking rules.
	origLang     bool
	origReg      bool
	regGroupDist uint8
	regDist      uint8
	origScript   bool
	parentDist   uint8 // 255 if have is not an ancestor of want tag.
}

// update updates the existing best match if the new pair is considered to be a
// better match.
// To determine if the given pair is a better match, it first computes the rough
// confidence level. If this surpasses the current match, it will replace it and
// update the tie-breaker rule cache. If there is a tie, it proceeds with applying
// a series of tie-breaker rules. If there is no conclusive winner after applying
// the tie-breaker rules, it leaves the current match as the preferred match.
func (m *bestMatch) update(have *haveTag, tag Tag, maxScript scriptID, maxRegion regionID) {
	// Bail if the maximum attainable confidence is below that of the current best match.
	c := have.conf
	if c < m.conf {
		return
	}
	if have.maxScript != maxScript {
		// There is usually very little comprehension between different scripts.
		// In a few cases there may still be Low comprehension. This possibility is
		// pre-computed and stored in have.altScript.
		if Low < m.conf || have.altScript != maxScript {
			return
		}
		c = Low
	} else if have.maxRegion != maxRegion {
		// There is usually a small difference between languages across regions.
		// We use the region distance (below) to disambiguate between equal matches.
		if High < c {
			c = High
		}
	}

	// We store the results of the computations of the tie-breaker rules along
	// with the best match. There is no need to do the checks once we determine
	// we have a winner, but we do still need to do the tie-breaker computations.
	// We use "beaten" to keep track if we still need to do the checks.
	beaten := false // true if the new pair defeats the current one.
	if c != m.conf {
		if c < m.conf {
			return
		}
		beaten = true
	}

	// Tie-breaker rules:
	// We prefer if the pre-maximized language was specified and identical.
	origLang := have.tag.lang == tag.lang && tag.lang != 0
	if !beaten && m.origLang != origLang {
		if m.origLang {
			return
		}
		beaten = true
	}

	regGroupDist := regionGroupDist(have.maxRegion, maxRegion, maxScript, tag.lang)
	if !beaten && m.regGroupDist != regGroupDist {
		if regGroupDist > m.regGroupDist {
			return
		}
		beaten = true
	}

	// We prefer if the pre-maximized region was specified and identical.
	origReg := have.tag.region == tag.region && tag.region != 0
	if !beaten && m.origReg != origReg {
		if m.origReg {
			return
		}
		beaten = true
	}

	// TODO: remove the region distance rule. Region distance has been replaced
	// by the region grouping rule. For now we leave it as it still seems to
	// have a net positive effect when applied after the grouping rule.
	// Possible solutions:
	// - apply the primary locale rule first to effectively disable region
	//   region distance if groups are defined.
	// - express the following errors in terms of grouping (if possible)
	// - find another method of handling the following cases.
	// maximization of legacy: find mo in
	//      "sr-Cyrl, sr-Latn, ro, ro-MD": have ro; want ro-MD (High)
	// region distance French: find fr-US in
	//      "en, fr, fr-CA, fr-CH": have fr; want fr-CA (High)

	// Next we prefer smaller distances between regions, as defined by
	// regionDist.
	regDist := uint8(regionDistance(have.maxRegion, maxRegion))
	if !beaten && m.regDist != regDist {
		if regDist > m.regDist {
			return
		}
		beaten = true
	}

	// Next we prefer if the pre-maximized script was specified and identical.
	origScript := have.tag.script == tag.script && tag.script != 0
	if !beaten && m.origScript != origScript {
		if m.origScript {
			return
		}
		beaten = true
	}

	// Finally we prefer tags which have a closer parent relationship.
	// TODO: the parent relationship no longer seems necessary. It doesn't hurt
	// to leave it in as the final tie-breaker, though, especially until the
	// grouping data has further matured.
	parentDist := parentDistance(have.tag.region, tag)
	if !beaten && m.parentDist != parentDist {
		if parentDist > m.parentDist {
			return
		}
		beaten = true
	}

	// Update m to the newly found best match.
	if beaten {
		m.have = have
		m.want = tag
		m.conf = c
		m.origLang = origLang
		m.origReg = origReg
		m.origScript = origScript
		m.regGroupDist = regGroupDist
		m.regDist = regDist
		m.parentDist = parentDist
	}
}

// parentDistance returns the number of times Parent must be called before the
// regions match. It is assumed that it has already been checked that lang and
// script are identical. If haveRegion does not occur in the ancestor chain of
// tag, it returns 255.
func parentDistance(haveRegion regionID, tag Tag) uint8 {
	p := tag.Parent()
	d := uint8(1)
	for haveRegion != p.region {
		if p.region == 0 {
			return 255
		}
		p = p.Parent()
		d++
	}
	return d
}

// regionGroupDist computes the distance between two regions based on their
// CLDR grouping.
func regionGroupDist(a, b regionID, script scriptID, lang langID) uint8 {
	aGroup := uint(regionToGroups[a]) << 1
	bGroup := uint(regionToGroups[b]) << 1
	for _, ri := range matchRegion {
		if langID(ri.lang) == lang && (ri.script == 0 || scriptID(ri.script) == script) {
			group := uint(1 << (ri.group &^ 0x80))
			if 0x80&ri.group == 0 {
				if aGroup&bGroup&group != 0 { // Both regions are in the group.
					return ri.distance
				}
			} else {
				if (aGroup|bGroup)&group == 0 { // Both regions are not in the group.
					return ri.distance
				}
			}
		}
	}
	const defaultDistance = 4
	return defaultDistance
}

// regionDistance computes the distance between two regions based on the
// distance in the graph of region containments as defined in CLDR. It iterates
// over increasingly inclusive sets of groups, represented as bit vectors, until
// the source bit vector has bits in common with the destination vector.
func regionDistance(a, b regionID) int {
	if a == b {
		return 0
	}
	p, q := regionInclusion[a], regionInclusion[b]
	if p < nRegionGroups {
		p, q = q, p
	}
	set := regionInclusionBits
	if q < nRegionGroups && set[p]&(1<<q) != 0 {
		return 1
	}
	d := 2
	for goal := set[q]; set[p]&goal == 0; p = regionInclusionNext[p] {
		d++
	}
	return d
}

func (t Tag) variants() string {
	if t.pVariant == 0 {
		return ""
	}
	return t.str[t.pVariant:t.pExt]
}

// variantOrPrivateTagStr returns variants or private use tags.
func (t Tag) variantOrPrivateTagStr() string {
	if t.pExt > 0 {
		return t.str[t.pVariant:t.pExt]
	}
	return t.str[t.pVariant:]
}

// equalsRest compares everything except the language.
func (a Tag) equalsRest(b Tag) bool {
	// TODO: don't include extensions in this comparison. To do this efficiently,
	// though, we should handle private tags separately.
	return a.script == b.script && a.region == b.region && a.variantOrPrivateTagStr() == b.variantOrPrivateTagStr()
}

// isExactEquivalent returns true if canonicalizing the language will not alter
// the script or region of a tag.
func isExactEquivalent(l langID) bool {
	for _, o := range notEquivalent {
		if o == l {
			return false
		}
	}
	return true
}

var notEquivalent []langID

func init() {
	// Create a list of all languages for which canonicalization may alter the
	// script or region.
	for _, lm := range langAliasMap {
		tag := Tag{lang: langID(lm.from)}
		if tag, _ = tag.canonicalize(All); tag.script != 0 || tag.region != 0 {
			notEquivalent = append(notEquivalent, langID(lm.from))
		}
	}
}
