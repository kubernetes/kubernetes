ASCII Table Writer
=========

[![Build Status](https://travis-ci.org/olekukonko/tablewriter.png?branch=master)](https://travis-ci.org/olekukonko/tablewriter)
[![Total views](https://img.shields.io/sourcegraph/rrc/github.com/olekukonko/tablewriter.svg)](https://sourcegraph.com/github.com/olekukonko/tablewriter)
[![Godoc](https://godoc.org/github.com/olekukonko/tablewriter?status.svg)](https://godoc.org/github.com/olekukonko/tablewriter)

Generate ASCII table on the fly ...  Installation is simple as

    go get github.com/olekukonko/tablewriter


#### Features
- Automatic Padding
- Support Multiple Lines
- Supports Alignment
- Support Custom Separators
- Automatic Alignment of numbers & percentage
- Write directly to http , file etc via `io.Writer`
- Read directly from CSV file
- Optional row line via `SetRowLine`
- Normalise table header
- Make CSV Headers optional
- Enable or disable table border
- Set custom footer support
- Optional identical cells merging
- Set custom caption
- Optional reflowing of paragraphs in multi-line cells.

#### Example   1 - Basic
```go
data := [][]string{
    []string{"A", "The Good", "500"},
    []string{"B", "The Very very Bad Man", "288"},
    []string{"C", "The Ugly", "120"},
    []string{"D", "The Gopher", "800"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Name", "Sign", "Rating"})

for _, v := range data {
    table.Append(v)
}
table.Render() // Send output
```

##### Output  1
```
+------+-----------------------+--------+
| NAME |         SIGN          | RATING |
+------+-----------------------+--------+
|  A   |       The Good        |    500 |
|  B   | The Very very Bad Man |    288 |
|  C   |       The Ugly        |    120 |
|  D   |      The Gopher       |    800 |
+------+-----------------------+--------+
```

#### Example 2 - Without Border / Footer / Bulk Append
```go
data := [][]string{
    []string{"1/1/2014", "Domain name", "2233", "$10.98"},
    []string{"1/1/2014", "January Hosting", "2233", "$54.95"},
    []string{"1/4/2014", "February Hosting", "2233", "$51.00"},
    []string{"1/4/2014", "February Extra Bandwidth", "2233", "$30.00"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Date", "Description", "CV2", "Amount"})
table.SetFooter([]string{"", "", "Total", "$146.93"}) // Add Footer
table.SetBorder(false)                                // Set Border to false
table.AppendBulk(data)                                // Add Bulk Data
table.Render()
```

##### Output 2
```

    DATE   |       DESCRIPTION        |  CV2  | AMOUNT
-----------+--------------------------+-------+----------
  1/1/2014 | Domain name              |  2233 | $10.98
  1/1/2014 | January Hosting          |  2233 | $54.95
  1/4/2014 | February Hosting         |  2233 | $51.00
  1/4/2014 | February Extra Bandwidth |  2233 | $30.00
-----------+--------------------------+-------+----------
                                        TOTAL | $146 93
                                      --------+----------

```


#### Example 3 - CSV
```go
table, _ := tablewriter.NewCSV(os.Stdout, "testdata/test_info.csv", true)
table.SetAlignment(tablewriter.ALIGN_LEFT)   // Set Alignment
table.Render()
```

##### Output 3
```
+----------+--------------+------+-----+---------+----------------+
|  FIELD   |     TYPE     | NULL | KEY | DEFAULT |     EXTRA      |
+----------+--------------+------+-----+---------+----------------+
| user_id  | smallint(5)  | NO   | PRI | NULL    | auto_increment |
| username | varchar(10)  | NO   |     | NULL    |                |
| password | varchar(100) | NO   |     | NULL    |                |
+----------+--------------+------+-----+---------+----------------+
```

#### Example 4  - Custom Separator
```go
table, _ := tablewriter.NewCSV(os.Stdout, "testdata/test.csv", true)
table.SetRowLine(true)         // Enable row line

// Change table lines
table.SetCenterSeparator("*")
table.SetColumnSeparator("╪")
table.SetRowSeparator("-")

table.SetAlignment(tablewriter.ALIGN_LEFT)
table.Render()
```

##### Output 4
```
*------------*-----------*---------*
╪ FIRST NAME ╪ LAST NAME ╪   SSN   ╪
*------------*-----------*---------*
╪ John       ╪ Barry     ╪ 123456  ╪
*------------*-----------*---------*
╪ Kathy      ╪ Smith     ╪ 687987  ╪
*------------*-----------*---------*
╪ Bob        ╪ McCornick ╪ 3979870 ╪
*------------*-----------*---------*
```

#### Example 5 - Markdown Format
```go
data := [][]string{
	[]string{"1/1/2014", "Domain name", "2233", "$10.98"},
	[]string{"1/1/2014", "January Hosting", "2233", "$54.95"},
	[]string{"1/4/2014", "February Hosting", "2233", "$51.00"},
	[]string{"1/4/2014", "February Extra Bandwidth", "2233", "$30.00"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Date", "Description", "CV2", "Amount"})
table.SetBorders(tablewriter.Border{Left: true, Top: false, Right: true, Bottom: false})
table.SetCenterSeparator("|")
table.AppendBulk(data) // Add Bulk Data
table.Render()
```

##### Output 5
```
|   DATE   |       DESCRIPTION        | CV2  | AMOUNT |
|----------|--------------------------|------|--------|
| 1/1/2014 | Domain name              | 2233 | $10.98 |
| 1/1/2014 | January Hosting          | 2233 | $54.95 |
| 1/4/2014 | February Hosting         | 2233 | $51.00 |
| 1/4/2014 | February Extra Bandwidth | 2233 | $30.00 |
```

#### Example 6  - Identical cells merging
```go
data := [][]string{
  []string{"1/1/2014", "Domain name", "1234", "$10.98"},
  []string{"1/1/2014", "January Hosting", "2345", "$54.95"},
  []string{"1/4/2014", "February Hosting", "3456", "$51.00"},
  []string{"1/4/2014", "February Extra Bandwidth", "4567", "$30.00"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Date", "Description", "CV2", "Amount"})
table.SetFooter([]string{"", "", "Total", "$146.93"})
table.SetAutoMergeCells(true)
table.SetRowLine(true)
table.AppendBulk(data)
table.Render()
```

##### Output 6
```
+----------+--------------------------+-------+---------+
|   DATE   |       DESCRIPTION        |  CV2  | AMOUNT  |
+----------+--------------------------+-------+---------+
| 1/1/2014 | Domain name              |  1234 | $10.98  |
+          +--------------------------+-------+---------+
|          | January Hosting          |  2345 | $54.95  |
+----------+--------------------------+-------+---------+
| 1/4/2014 | February Hosting         |  3456 | $51.00  |
+          +--------------------------+-------+---------+
|          | February Extra Bandwidth |  4567 | $30.00  |
+----------+--------------------------+-------+---------+
|                                       TOTAL | $146 93 |
+----------+--------------------------+-------+---------+
```

#### Example 7  - Identical cells merging (specify the column index to merge)
```go
data := [][]string{
  []string{"1/1/2014", "Domain name", "1234", "$10.98"},
  []string{"1/1/2014", "January Hosting", "1234", "$10.98"},
  []string{"1/4/2014", "February Hosting", "3456", "$51.00"},
  []string{"1/4/2014", "February Extra Bandwidth", "4567", "$30.00"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Date", "Description", "CV2", "Amount"})
table.SetFooter([]string{"", "", "Total", "$146.93"})
table.SetAutoMergeCellsByColumnIndex([]int{2, 3})
table.SetRowLine(true)
table.AppendBulk(data)
table.Render()
```

##### Output 7
```
+----------+--------------------------+-------+---------+
|   DATE   |       DESCRIPTION        |  CV2  | AMOUNT  |
+----------+--------------------------+-------+---------+
| 1/1/2014 | Domain name              |  1234 | $10.98  |
+----------+--------------------------+       +         +
| 1/1/2014 | January Hosting          |       |         |
+----------+--------------------------+-------+---------+
| 1/4/2014 | February Hosting         |  3456 | $51.00  |
+----------+--------------------------+-------+---------+
| 1/4/2014 | February Extra Bandwidth |  4567 | $30.00  |
+----------+--------------------------+-------+---------+
|                                       TOTAL | $146.93 |
+----------+--------------------------+-------+---------+
```


#### Table with color
```go
data := [][]string{
	[]string{"1/1/2014", "Domain name", "2233", "$10.98"},
	[]string{"1/1/2014", "January Hosting", "2233", "$54.95"},
	[]string{"1/4/2014", "February Hosting", "2233", "$51.00"},
	[]string{"1/4/2014", "February Extra Bandwidth", "2233", "$30.00"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Date", "Description", "CV2", "Amount"})
table.SetFooter([]string{"", "", "Total", "$146.93"}) // Add Footer
table.SetBorder(false)                                // Set Border to false

table.SetHeaderColor(tablewriter.Colors{tablewriter.Bold, tablewriter.BgGreenColor},
	tablewriter.Colors{tablewriter.FgHiRedColor, tablewriter.Bold, tablewriter.BgBlackColor},
	tablewriter.Colors{tablewriter.BgRedColor, tablewriter.FgWhiteColor},
	tablewriter.Colors{tablewriter.BgCyanColor, tablewriter.FgWhiteColor})

table.SetColumnColor(tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiBlackColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiRedColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiBlackColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgBlackColor})

table.SetFooterColor(tablewriter.Colors{}, tablewriter.Colors{},
	tablewriter.Colors{tablewriter.Bold},
	tablewriter.Colors{tablewriter.FgHiRedColor})

table.AppendBulk(data)
table.Render()
```

#### Table with color Output
![Table with Color](https://cloud.githubusercontent.com/assets/6460392/21101956/bbc7b356-c0a1-11e6-9f36-dba694746efc.png)

#### Example - 8 Table Cells with Color

Individual Cell Colors from `func Rich` take precedence over Column Colors

```go
data := [][]string{
	[]string{"Test1Merge", "HelloCol2 - 1", "HelloCol3 - 1", "HelloCol4 - 1"},
	[]string{"Test1Merge", "HelloCol2 - 2", "HelloCol3 - 2", "HelloCol4 - 2"},
	[]string{"Test1Merge", "HelloCol2 - 3", "HelloCol3 - 3", "HelloCol4 - 3"},
	[]string{"Test2Merge", "HelloCol2 - 4", "HelloCol3 - 4", "HelloCol4 - 4"},
	[]string{"Test2Merge", "HelloCol2 - 5", "HelloCol3 - 5", "HelloCol4 - 5"},
	[]string{"Test2Merge", "HelloCol2 - 6", "HelloCol3 - 6", "HelloCol4 - 6"},
	[]string{"Test2Merge", "HelloCol2 - 7", "HelloCol3 - 7", "HelloCol4 - 7"},
	[]string{"Test3Merge", "HelloCol2 - 8", "HelloCol3 - 8", "HelloCol4 - 8"},
	[]string{"Test3Merge", "HelloCol2 - 9", "HelloCol3 - 9", "HelloCol4 - 9"},
	[]string{"Test3Merge", "HelloCol2 - 10", "HelloCol3 -10", "HelloCol4 - 10"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Col1", "Col2", "Col3", "Col4"})
table.SetFooter([]string{"", "", "Footer3", "Footer4"})
table.SetBorder(false)

table.SetHeaderColor(tablewriter.Colors{tablewriter.Bold, tablewriter.BgGreenColor},
	tablewriter.Colors{tablewriter.FgHiRedColor, tablewriter.Bold, tablewriter.BgBlackColor},
	tablewriter.Colors{tablewriter.BgRedColor, tablewriter.FgWhiteColor},
	tablewriter.Colors{tablewriter.BgCyanColor, tablewriter.FgWhiteColor})

table.SetColumnColor(tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiBlackColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiRedColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgHiBlackColor},
	tablewriter.Colors{tablewriter.Bold, tablewriter.FgBlackColor})

table.SetFooterColor(tablewriter.Colors{}, tablewriter.Colors{},
	tablewriter.Colors{tablewriter.Bold},
	tablewriter.Colors{tablewriter.FgHiRedColor})

colorData1 := []string{"TestCOLOR1Merge", "HelloCol2 - COLOR1", "HelloCol3 - COLOR1", "HelloCol4 - COLOR1"}
colorData2 := []string{"TestCOLOR2Merge", "HelloCol2 - COLOR2", "HelloCol3 - COLOR2", "HelloCol4 - COLOR2"}

for i, row := range data {
	if i == 4 {
		table.Rich(colorData1, []tablewriter.Colors{tablewriter.Colors{}, tablewriter.Colors{tablewriter.Normal, tablewriter.FgCyanColor}, tablewriter.Colors{tablewriter.Bold, tablewriter.FgWhiteColor}, tablewriter.Colors{}})
		table.Rich(colorData2, []tablewriter.Colors{tablewriter.Colors{tablewriter.Normal, tablewriter.FgMagentaColor}, tablewriter.Colors{}, tablewriter.Colors{tablewriter.Bold, tablewriter.BgRedColor}, tablewriter.Colors{tablewriter.FgHiGreenColor, tablewriter.Italic, tablewriter.BgHiCyanColor}})
	}
	table.Append(row)
}

table.SetAutoMergeCells(true)
table.Render()

```

##### Table cells with color Output
![Table cells with Color](https://user-images.githubusercontent.com/9064687/63969376-bcd88d80-ca6f-11e9-9466-c3d954700b25.png)

#### Example 9 - Set table caption
```go
data := [][]string{
    []string{"A", "The Good", "500"},
    []string{"B", "The Very very Bad Man", "288"},
    []string{"C", "The Ugly", "120"},
    []string{"D", "The Gopher", "800"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Name", "Sign", "Rating"})
table.SetCaption(true, "Movie ratings.")

for _, v := range data {
    table.Append(v)
}
table.Render() // Send output
```

Note: Caption text will wrap with total width of rendered table.

##### Output 9
```
+------+-----------------------+--------+
| NAME |         SIGN          | RATING |
+------+-----------------------+--------+
|  A   |       The Good        |    500 |
|  B   | The Very very Bad Man |    288 |
|  C   |       The Ugly        |    120 |
|  D   |      The Gopher       |    800 |
+------+-----------------------+--------+
Movie ratings.
```

#### Example 10 - Set NoWhiteSpace and TablePadding option
```go
data := [][]string{
    {"node1.example.com", "Ready", "compute", "1.11"},
    {"node2.example.com", "Ready", "compute", "1.11"},
    {"node3.example.com", "Ready", "compute", "1.11"},
    {"node4.example.com", "NotReady", "compute", "1.11"},
}

table := tablewriter.NewWriter(os.Stdout)
table.SetHeader([]string{"Name", "Status", "Role", "Version"})
table.SetAutoWrapText(false)
table.SetAutoFormatHeaders(true)
table.SetHeaderAlignment(ALIGN_LEFT)
table.SetAlignment(ALIGN_LEFT)
table.SetCenterSeparator("")
table.SetColumnSeparator("")
table.SetRowSeparator("")
table.SetHeaderLine(false)
table.SetBorder(false)
table.SetTablePadding("\t") // pad with tabs
table.SetNoWhiteSpace(true)
table.AppendBulk(data) // Add Bulk Data
table.Render()
```

##### Output 10
```
NAME             	STATUS  	ROLE   	VERSION 
node1.example.com	Ready   	compute	1.11   	
node2.example.com	Ready   	compute	1.11   	
node3.example.com	Ready   	compute	1.11   	
node4.example.com	NotReady	compute	1.11   	
```

#### Render table into a string

Instead of rendering the table to `io.Stdout` you can also render it into a string. Go 1.10 introduced the `strings.Builder` type which implements the `io.Writer` interface and can therefore be used for this task. Example:

```go
package main

import (
    "strings"
    "fmt"

    "github.com/olekukonko/tablewriter"
)

func main() {
    tableString := &strings.Builder{}
    table := tablewriter.NewWriter(tableString)

    /*
     * Code to fill the table
     */

    table.Render()

    fmt.Println(tableString.String())
}
```

#### TODO
- ~~Import Directly from CSV~~  - `done`
- ~~Support for `SetFooter`~~  - `done`
- ~~Support for `SetBorder`~~  - `done`
- ~~Support table with uneven rows~~ - `done`
- ~~Support custom alignment~~
- General Improvement & Optimisation
- `NewHTML` Parse table from HTML
