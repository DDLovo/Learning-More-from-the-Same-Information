"""Materialize fixed study-authored ECG prompts and their optional study rubric."""

import argparse
from pathlib import Path

from . import CLASSES
from .teacher import validate_questions, write_json


MANUAL = (
    ('On a clear adult lead II tracing in <object>, what usually describes the discrete atrial deflections?', (
        'No reproducible discrete P waves',
        'Upright P waves of broadly consistent shape when visible',
        'Inverted P waves of consistent shape',
        'P waves cycling through several clearly different shapes',
        'Atrial deflections fixed to the trailing edge of QRS complexes',
    )),
    ('With intact atrioventricular conduction and no isolated ectopic beats, what pattern of successive R-R intervals is typical of <object>?', (
        'Alternating one short interval and one long interval',
        'Intervals repeating in a fixed pattern of three beats',
        'Unpredictably irregular intervals without a repeating pattern',
        'Predominantly regular intervals, possibly with mild respiratory variation',
        'Intervals lengthening progressively over the entire tracing',
    )),
    ('For a typical resting adult with intact conduction, which ventricular rate category characterizes <object> when the unqualified organized-rhythm label denotes the ordinary rate range?', (
        'Typically over 100 beats per minute',
        'No particular ventricular rate range is specified by this rhythm label',
        'Repeated alternation between rates below 60 and above 100 beats per minute is required',
        'Typically below 60 beats per minute',
        'Typically in the 60 to 100 beats per minute range',
    )),
    ('What does the atrial baseline usually look like between identifiable ventricular complexes in <object> on a low-noise tracing?', (
        'A repeating sawtooth pattern throughout the ventricular cycle',
        'An isoelectric segment apart from discrete organized deflections',
        'A broad ventricular deflection throughout the interval',
        'A fixed-amplitude oscillation locked to a single frequency',
        'Irregular atrial activity in place of organized P waves, sometimes subtle',
    )),
    ('When atrioventricular conduction is intact, which relationship between discrete P waves and QRS complexes usually occurs in <object>?', (
        'Two organized P waves precede each QRS complex',
        'P waves and QRS complexes recur independently',
        'One organized P wave precedes each QRS complex',
        'No discrete P wave is consistently available to pair with each QRS complex',
        'One organized P wave follows each QRS complex',
    )),
    ('Without assuming a separate conduction abnormality, how is a PR interval characterized in <object>?', (
        'A consistent PR interval longer than 200 milliseconds is required',
        'There is no discrete P wave from which to measure a standard PR interval',
        'The PR interval must shorten by a fixed amount every beat',
        'PR duration must be below 120 milliseconds',
        'A PR interval can usually be measured on clear beats, without one required duration',
    )),
    ('Which kind of evidence about atrial organization would you most commonly find in <object> with good signal quality?', (
        'Organized atrial activations with broadly similar morphology and a measurable cycle length',
        'A regular series of several atrial activations between every pair of ventricular beats',
        'No stable beat-to-beat cycle of discrete atrial activations',
        'A repeating sequence of three different atrial morphologies',
        'Isolated atrial activations only after unusually long pauses',
    )),
    ('How is the typical time spacing of QRS complexes in <object> described on a clear tracing with intact atrioventricular conduction?', (
        'Beat spacing varies irregularly and its average is not fixed by the rhythm',
        'Relatively regular spacing, typically between 0.6 and 1.0 seconds',
        'An alternating short-long spacing pattern',
        'Relatively regular spacing, typically longer than 1.0 second',
        'Relatively regular spacing, typically shorter than 0.6 seconds',
    )),
    ('When comparing simultaneous lead II and V1 in <object>, what generally holds for the timing of identifiable P waves?', (
        'Lead V1 has twice as many organized atrial activation events as lead II',
        'The atrial activation is identifiable only at exactly the end of QRS in both leads',
        'The same organized atrial activation is represented at approximately the same time across leads where P is visible',
        'Organized atrial activations occur at unrelated times in different leads',
        'There is no stable series of discrete P waves to align across leads',
    )),
    ('If the expected number of QRS complexes in 30 seconds is estimated by tripling the mean count in 10-second strips of stable <object> with intact conduction, which range applies?', (
        'More than 50 complexes per 30 seconds on average',
        'Between 30 and 50 complexes per 30 seconds on average, inclusive',
        'Each strip must alternate between the lower and upper ranges',
        'Fewer than 30 complexes per 30 seconds on average',
        'No single count range follows from this rhythm label',
    )),
    ('If typical beats of <object> without ectopy are aligned on R peaks, what happens to the pre-QRS atrial deflection when beats are averaged?', (
        'Two distinct upright atrial deflections consistently emerge per beat',
        'A reproducible pre-QRS P wave is often retained when conduction is stable',
        'A high-amplitude sawtooth waveform is recovered before every QRS',
        'No stable pre-QRS P wave is expected to emerge from the average',
        'Each averaged complex requires an atrial wave immediately following its R peak',
    )),
    ('Which statement about QRS duration follows from the rhythm name <object> alone, when separate conduction findings have not been specified?', (
        'QRS duration must exceed 120 milliseconds',
        'QRS duration must be below 80 milliseconds',
        'QRS duration must alternate between wide and narrow from beat to beat',
        'The rhythm name leaves QRS duration unspecified',
        'Every QRS duration must be exactly 100 milliseconds',
    )),
)

ASTRA = (
    ('Which finding is characteristic of <object> when the mean ventricular rate is estimated from a resting ECG?', (
        'Above 100 beats/min',
        '60–100 beats/min, inclusive',
        'No defining rate band; slow, ordinary, and fast rates are all possible',
        'Below 60 beats/min',
        'No measurable ventricular rate because QRS complexes are absent',
    )),
    ('Which sequence of R–R intervals is typical of <object>?', (
        'A fixed short–long alternation',
        'Repeating groups of progressively shorter intervals followed by a pause',
        'Two steady spacings with one abrupt transition between them',
        'Nearly equal intervals, possibly with slight, smoothly cyclic variation',
        'Irregular spacing without a repeating pattern',
    )),
    ('Which pattern of P-wave shapes is typical of <object> when successive beats are examined in the same lead?', (
        'One consistent shape',
        'Exactly two shapes in strict alternation',
        'Exactly two shapes in no fixed order',
        'Three or more distinct, recurring shapes',
        'No consistently identifiable discrete P waves',
    )),
    ('Which P-wave polarity pattern is typical in lead II during <object>?', (
        'Entirely negative',
        'Biphasic, with a positive then a negative component',
        'No discrete P wave to which polarity can be assigned',
        'Upright',
        'Biphasic, with a negative then a positive component',
    )),
    ('What count of discrete P waves per QRS complex is typical across complete cycles in <object>?', (
        'Exactly two per QRS complex',
        'No countable discrete P waves',
        'One per QRS complex',
        'Exactly three per QRS complex',
        'One per pair of QRS complexes',
    )),
    ('In a lead that displays atrial activity well, what pattern does that activity typically form between ventricular deflections in <object>?', (
        'Continuous, regular sawtooth deflections',
        'Repeating pairs of atrial waves separated by quiet intervals',
        'Repeating bursts of three or more atrial waves separated by quiet intervals',
        'Irregular undulations that may be very subtle, without discrete P waves',
        'Individual P waves with an otherwise quiet atrial baseline',
    )),
    ('Which result is typical when the mean interval between successive discrete P-wave onsets is estimated in <object>?', (
        'Longer than 1,000 ms',
        'Shorter than 250 ms',
        'Not measurable because discrete P-wave onsets cannot be identified',
        '600–1,000 ms, inclusive',
        'At least 250 ms but shorter than 600 ms',
    )),
    ('Where is a discrete P wave typically found relative to its associated QRS complex in <object>?', (
        'Within the QRS complex',
        'Before QRS onset',
        'In the ST segment following the QRS complex',
        'Within the T wave following the QRS complex',
        'No discrete P wave can be assigned to the complex',
    )),
    ('At a paper speed of 25 mm/s, what mean horizontal spacing between successive QRS onsets is characteristic of <object>?', (
        'No defining spacing band; the mean depends on the ventricular response',
        'More than five large squares',
        'Fewer than one and a half large squares',
        'Three to five large squares, inclusive',
        'At least one and a half but fewer than three large squares',
    )),
    ('What is typically found when the PR interval is compared across successive beats in <object>?', (
        'Progressive lengthening in repeating groups',
        'Strict alternation between two distinct values',
        'Little change in a measurable interval',
        'Irregular variation among clearly measurable intervals',
        'No measurable interval because a discrete P-wave onset is lacking',
    )),
    ('If successive lead II beats in <object> are superimposed at QRS onset, what atrial pattern is typically seen just before that alignment point?', (
        'A discrete P-wave contour shifting progressively in time on successive beats',
        'No aligned P-wave contour emerging from the overlaid baselines',
        'Two distinct P-wave contours appearing on alternate beats',
        'A train of several regularly spaced atrial waves aligned across beats',
        'A single P-wave contour recurring at nearly the same position',
    )),
    ('Which conclusion about QRS duration is warranted by the label <object> alone?', (
        'It must be shorter than 80 ms',
        'It must be at least 120 ms but shorter than 160 ms',
        'It must be at least 160 ms',
        'No particular QRS duration is required',
        'It must be at least 80 ms but shorter than 120 ms',
    )),
)

ASTRA_REFERENCES = (
    ('B', 'C', 'A', 'D'),
    ('D', 'E', 'D', 'D'),
    ('A', 'E', 'A', 'A'),
    ('D', 'C', 'D', 'D'),
    ('C', 'B', 'C', 'C'),
    ('E', 'D', 'E', 'E'),
    ('D', 'C', 'E', 'A'),
    ('B', 'E', 'B', 'B'),
    ('D', 'A', 'E', 'B'),
    ('C', 'E', 'C', 'C'),
    ('E', 'B', 'E', 'E'),
    ('D', 'D', 'D', 'D'),
)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", choices=("manual", "astra"), default="astra")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rubric", type=Path)
    args = parser.parse_args()
    if args.rubric and args.bank != "astra":
        parser.error("The reference rubric is available for the Astra bank.")
    if args.rubric and args.rubric.resolve() == args.out.resolve():
        parser.error("Use separate question and rubric output paths.")
    selected = ASTRA if args.bank == "astra" else MANUAL
    questions = [{"question": question, "options": list(options)} for question, options in selected]
    write_json(args.out, validate_questions(questions, expected_count=12))
    if args.rubric:
        write_json(args.rubric, {"class_codes": list(CLASSES), "expected_options_by_question": ASTRA_REFERENCES})


if __name__ == "__main__":
    main()
