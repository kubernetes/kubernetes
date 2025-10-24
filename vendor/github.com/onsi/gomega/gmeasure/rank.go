package gmeasure

import (
	"fmt"
	"sort"

	"github.com/onsi/gomega/gmeasure/table"
)

/*
RankingCriteria is an enum representing the criteria by which Stats should be ranked.  The enum names should be self explanatory.  e.g. LowerMeanIsBetter means that Stats with lower mean values are considered more beneficial, with the lowest mean being declared the "winner" .
*/
type RankingCriteria uint

const (
	LowerMeanIsBetter RankingCriteria = iota
	HigherMeanIsBetter
	LowerMedianIsBetter
	HigherMedianIsBetter
	LowerMinIsBetter
	HigherMinIsBetter
	LowerMaxIsBetter
	HigherMaxIsBetter
)

var rcEnumSupport = newEnumSupport(map[uint]string{uint(LowerMeanIsBetter): "Lower Mean is Better", uint(HigherMeanIsBetter): "Higher Mean is Better", uint(LowerMedianIsBetter): "Lower Median is Better", uint(HigherMedianIsBetter): "Higher Median is Better", uint(LowerMinIsBetter): "Lower Mins is Better", uint(HigherMinIsBetter): "Higher Min is Better", uint(LowerMaxIsBetter): "Lower Max is Better", uint(HigherMaxIsBetter): "Higher Max is Better"})

func (s RankingCriteria) String() string { return rcEnumSupport.String(uint(s)) }
func (s *RankingCriteria) UnmarshalJSON(b []byte) error {
	out, err := rcEnumSupport.UnmarshJSON(b)
	*s = RankingCriteria(out)
	return err
}
func (s RankingCriteria) MarshalJSON() ([]byte, error) { return rcEnumSupport.MarshJSON(uint(s)) }

/*
Ranking ranks a set of Stats by a specified RankingCriteria.  Use RankStats to create a Ranking.

When using Ginkgo, you can register Rankings as Report Entries via AddReportEntry.  This will emit a formatted table representing the Stats in rank-order when Ginkgo generates the report.
*/
type Ranking struct {
	Criteria RankingCriteria
	Stats    []Stats
}

/*
RankStats creates a new ranking of the passed-in stats according to the passed-in criteria.
*/
func RankStats(criteria RankingCriteria, stats ...Stats) Ranking {
	sort.Slice(stats, func(i int, j int) bool {
		switch criteria {
		case LowerMeanIsBetter:
			return stats[i].FloatFor(StatMean) < stats[j].FloatFor(StatMean)
		case HigherMeanIsBetter:
			return stats[i].FloatFor(StatMean) > stats[j].FloatFor(StatMean)
		case LowerMedianIsBetter:
			return stats[i].FloatFor(StatMedian) < stats[j].FloatFor(StatMedian)
		case HigherMedianIsBetter:
			return stats[i].FloatFor(StatMedian) > stats[j].FloatFor(StatMedian)
		case LowerMinIsBetter:
			return stats[i].FloatFor(StatMin) < stats[j].FloatFor(StatMin)
		case HigherMinIsBetter:
			return stats[i].FloatFor(StatMin) > stats[j].FloatFor(StatMin)
		case LowerMaxIsBetter:
			return stats[i].FloatFor(StatMax) < stats[j].FloatFor(StatMax)
		case HigherMaxIsBetter:
			return stats[i].FloatFor(StatMax) > stats[j].FloatFor(StatMax)
		}
		return false
	})

	out := Ranking{
		Criteria: criteria,
		Stats:    stats,
	}

	return out
}

/*
Winner returns the Stats with the most optimal rank based on the specified ranking criteria.  For example, if the RankingCriteria is LowerMaxIsBetter then the Stats with the lowest value or duration for StatMax will be returned as the "winner"
*/
func (c Ranking) Winner() Stats {
	if len(c.Stats) == 0 {
		return Stats{}
	}
	return c.Stats[0]
}

func (c Ranking) report(enableStyling bool) string {
	if len(c.Stats) == 0 {
		return "Empty Ranking"
	}
	t := table.NewTable()
	t.TableStyle.EnableTextStyling = enableStyling
	t.AppendRow(table.R(
		table.C("Experiment"), table.C("Name"), table.C("N"), table.C("Min"), table.C("Median"), table.C("Mean"), table.C("StdDev"), table.C("Max"),
		table.Divider("="),
		"{{bold}}",
	))

	for idx, stats := range c.Stats {
		name := stats.MeasurementName
		if stats.Units != "" {
			name = name + " [" + stats.Units + "]"
		}
		experimentName := stats.ExperimentName
		style := stats.Style
		if idx == 0 {
			style = "{{bold}}" + style
			name += "\n*Winner*"
			experimentName += "\n*Winner*"
		}
		r := table.R(style)
		t.AppendRow(r)
		r.AppendCell(table.C(experimentName), table.C(name))
		r.AppendCell(stats.cells()...)

	}
	out := fmt.Sprintf("Ranking Criteria: %s\n", c.Criteria)
	if enableStyling {
		out = "{{bold}}" + out + "{{/}}"
	}
	out += t.Render()
	return out
}

/*
ColorableString generates a styled report that includes a table of the rank-ordered Stats
It is called automatically by Ginkgo's reporting infrastructure when the Ranking is registered as a ReportEntry via AddReportEntry.
*/
func (c Ranking) ColorableString() string {
	return c.report(true)
}

/*
String generates an unstyled report that includes a table of the rank-ordered Stats
*/
func (c Ranking) String() string {
	return c.report(false)
}
