package main

import (
	"fmt"
	"strings"
	"time"
)

type ProgressBar struct {
	total      int
	current    int
	width      int
	fillChar   string
	emptyChar  string
	percentage int
	startTime  time.Time
}

func NewProgressBar(total, width int) *ProgressBar {
	return &ProgressBar{
		total:     total,
		width:     width,
		fillChar:  "█",
		emptyChar: "░",
		startTime: time.Now(),
	}
}

func (pb *ProgressBar) Increment() {
	pb.current++
	pb.percentage = int(float64(pb.current) / float64(pb.total) * 100)
	fmt.Print(pb.Render())
}

func (pb *ProgressBar) Render() string {
	filled := int(float64(pb.width) * float64(pb.current) / float64(pb.total))
	bar := strings.Repeat(pb.fillChar, filled) + strings.Repeat(pb.emptyChar, pb.width-filled)

	elapsed := time.Since(pb.startTime)
	var eta time.Duration
	var samplesPerSecond float64

	if pb.current > 0 {
		eta = time.Duration(float64(elapsed) * float64(pb.total-pb.current) / float64(pb.current)).Round(time.Second)
		samplesPerSecond = float64(pb.current) / elapsed.Seconds()
	}

	return fmt.Sprintf("\r %d%% [%s] (%d/%d, %.0f it/s) | %s | ETA: %s       ",
		pb.percentage, bar, pb.current, pb.total, samplesPerSecond,
		elapsed.Round(time.Second), eta.Round(time.Second))
}
