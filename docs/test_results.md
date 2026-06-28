# Squat Form Analyser - Test Results

## Testing Methodology

This document records the testing performed on the squat form analyser system. Testing was conducted using multiple sources to ensure comprehensive evaluation while maintaining ethical standards.

### Test Sources
1. **Self-testing** - Baseline testing by the author
2. **YouTube videos** - Publicly available instructional and demonstration videos
3. **AI-generated videos** - Controlled test scenarios using AI video generation tools
4. **Stock fitness videos** - Diverse demographic testing from Pexels

### Ethical Considerations
- No participants were recruited
- All human videos used were publicly available
- AI-generated content was used to test edge cases
- No personal data was collected or stored

### Metrics Recorded
- **Correct classifications** - System matches expected outcome
- **False positives** - System flags bad form when form is good
- **False negatives** - System misses bad form
- **Rep counter accuracy** - Reps counted correctly

---

## Test 1: Self-Testing (Author)

### Test 1.1 - Good Form Front Squats
**Date:** 
**Conditions:** 
**Reps:** 10

**Results:**
- Correct classifications: 10/10
- False positives (knees_in): 0
- Reps counted: 10/10
- Notes: 

### Test 1.2 - Knees Caved In (Deliberate Bad Form)
**Date:** 
**Conditions:** 
**Reps:** 10

**Results:**
- Correct classifications: 3/3
- False negatives: 0
- Reps counted: 3/3
- Notes: 

### Test 1.3 - Good Form Side Squats
**Date:** 
**Conditions:** 
**Reps:** 10

**Results:**
- Correct classifications: 10/10
- False positives (leaning_forward): 0
- Reps counted: 10/10
- Notes: 

### Test 1.4 - Leaning Forward (Deliberate Bad Form)
**Date:** 
**Conditions:** 
**Reps:** 10

**Results:**
- Correct classifications: 3/3
- False negatives: 0
- Reps counted: 3/3
- Notes: 

---

## Test 2: YouTube + AI-Generated Test Videos

A set of 37 pre-recorded test videos (sourced from YouTube and AI-generated clips) was organised into folders by expected form type and run through the system. Source URLs to be added.

**Results by folder:**

| Folder | Expected | Correct | Accuracy |
|--------|----------|---------|----------|
| front  | good (front) | 14/14 | 100% |
| side   | good (side)  | 12/16 | 75% |
| knees  | knees_in     | 0/2   | 0% |
| lean   | leaning_forward | 3/5 | 60% |
| **Overall** | | **29/37** | **78.4%** |

**Source URLs:** _(to be added)_

**Notes:** 

---

## Test 2b: Retest After Lean Model Improvements

The set of test videos was expanded to 43 and re-run after improving the side/lean model. The original trunk-angle feature was replaced with a vertical trunk angle (torso tilt from upright) and the side model retrained, and a confidence-based leaning rule was added (a video is classified as leaning when a confident share of side frames shows lean, rather than a simple majority vote).

**Results by folder:**

| Folder | Expected | Correct | Accuracy |
|--------|----------|---------|----------|
| front  | good (front) | 15/15 | 100% |
| side   | good (side)  | 16/16 | 100% |
| knees  | knees_in     | 1/6   | 16.7% |
| lean   | leaning_forward | 4/6 | 66.7% |
| **Overall** | | **36/43** | **83.7%** |

**Source URLs:** _(to be added)_

**Notes:** Side good-form classification improved from 75% to 100% and leaning detection from 60% to 66.7%, raising overall accuracy from 78.4% to 83.7%. Knees_in remains a known limitation (only sustained, deep knee collapse is detectable from a front-on 2D camera).

---

## Test 3: Environmental Variations

### Test 3.1 - Different Lighting
**Conditions tested:** 
**Results:** 

### Test 3.2 - Different Distances
**Conditions tested:** 
**Results:** 

### Test 3.3 - Different Clothing
**Conditions tested:** 
**Results:** 

---

## Summary

### Overall Performance
- Total tests conducted: 43
- Overall accuracy: 29/37 (83.7.%)
- Front model accuracy: 14/14 (100%)
- Side model accuracy: 15/21 (good 12/16, leaning 3/5)
- Rep counter accuracy: 10/10 front, 10/10 side (live self-test)

### Strengths
- Front-view rep counting and classification reliable (100% on test set)
- Live real-time operation confirmed (correct counting and fault detection)
- Side-view good-form classification strong at proper side angles

### Limitations Discovered
- Leaning detection limited: trunk-angle feature nearly identical for good vs leaning squats
- Side-view degrades at oblique camera angles (2D projection is view-dependent)
- Knees_in not detected on test clips (under investigation)

### Recommended Improvements
- Replace trunk-angle feature with vertical trunk angle (torso tilt from upright) and retrain
- Add camera-positioning guidance for users
- Expand training data with more oblique-angle and leaning examples