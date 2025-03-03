package progress_bar

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
	ticker     *time.Ticker
	stopChan   chan struct{}
}

func New(total, width int) *ProgressBar {
	pb := &ProgressBar{
		total:     total,
		width:     width,
		fillChar:  "█",
		emptyChar: "░",
		startTime: time.Now(),
		stopChan:  make(chan struct{}),
	}

	// Start the auto-refresh automatically
	pb.startTicker()

	return pb
}

func (pb *ProgressBar) startTicker() {
	pb.ticker = time.NewTicker(1 * time.Second)

	go func() {
		// Print immediately on start
		pb.Print()

		for {
			select {
			case <-pb.ticker.C:
				pb.Print()
			case <-pb.stopChan:
				pb.ticker.Stop()
				return
			}
		}
	}()
}

func (pb *ProgressBar) stopTicker() {
	pb.stopChan <- struct{}{}
}

func (pb *ProgressBar) Increment() {
	pb.current++

	// If we've reached the total, print the final state and stop the ticker
	if pb.current >= pb.total {
		pb.current = pb.total // Ensure we don't exceed total
		pb.Print()            // Print final state immediately
		fmt.Println()         // Add newline to finalize output
		pb.stopTicker()       // Stop the ticker
	}
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

	pb.percentage = int(float64(pb.current) / float64(pb.total) * 100)

	return fmt.Sprintf("\r %d%% [%s] (%d/%d, %.0f it/s) | %s | ETA: %s       ",
		pb.percentage, bar, pb.current, pb.total, samplesPerSecond,
		elapsed.Round(time.Second), eta.Round(time.Second))
}

func (pb *ProgressBar) Print() {
	fmt.Print(pb.Render())
}

// ForceComplete forces the progress bar to complete, regardless of current count
func (pb *ProgressBar) ForceComplete() {
	pb.current = pb.total
	pb.Print()
	fmt.Println()
	pb.stopTicker()
}
