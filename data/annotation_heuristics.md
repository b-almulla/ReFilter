# Annotation Heuristics for Functional Similarity

This document describes the annotation heuristics used to construct the manually labeled ground-truth data for ReFilter. The goal of annotation is to determine whether a candidate app is functionally similar to a target app.

## Core Definition

A candidate app is considered **functionally similar** to a target app if a user could reasonably select it as an alternative to the target app without losing the target app's core functionality.

In other words, annotators judge whether the candidate app satisfies the same primary user need as the target app, rather than whether the two apps merely share keywords, categories, interface styles, or target audiences.

## General Annotation Procedure

### Step 1: Identify the Target App's Primary Functional Purpose

Annotators first identify the target app's main user-facing purpose. This is the primary problem the app is designed to solve or the main task it enables.

Examples of primary purposes include:

- teaching Spanish;
- recording snoring for sleep monitoring;
- providing horoscope, tarot, numerology, or guidance-related predictions;
- supporting non-verbal communication through text-to-speech;
- managing passwords;
- providing offline maps;
- providing backing tracks for music practice.

### Step 2: Evaluate Each Candidate Against That Purpose

For each target-candidate pair, annotators assessed whether the candidate app meaningfully supports the same primary functional purpose as the target app.

A candidate app **does not need to match every feature** of the target app. It can be labeled as similar if it satisfies the same core user need, even when that functionality appears as a secondary or supporting feature rather than the candidate app's dominant purpose.

A candidate app should be labeled as **not similar** when it shares only:

- broad app-store category;
- keywords in the description;
- visual interface style;
- target audience;
- secondary features that do not satisfy the target app's core purpose.

## Use of Descriptions and Screenshots

Annotators primarily relied on app descriptions. If the app description did not clearly mention a relevant feature but the screenshots showed that the feature existed, annotators could use the screenshot evidence.

## Directionality

Functional similarity was judged from the perspective of the **target app**. The question was whether the candidate app could serve as an alternative for the target app's primary purpose. The relationship did not need to be strictly bidirectional.

## Edge-Case Rules and Examples

The following examples illustrate how the annotation rules were applied.

| Target app purpose | Similar candidates | Not similar candidates |
|---|---|---|
| Teaching Spanish | Apps that teach Spanish; apps that teach multiple languages including Spanish | Translation apps that translate Spanish but do not teach it |
| Recording snoring | Apps that record snoring, even if snoring is not the app's main feature | Apps that only provide sleep remedies or sleep sounds without snore recording |
| Text-to-speech communication for Mute users | Apps with text-to-speech functionality, even if not designed specifically for mute users | Translation apps or speech-related tools without text-to-speech communication |
| Password management | Apps with password management functionality | General note-taking, wallet, or list-management apps without password management |
| Watching live soccer matches | Apps that allow users to watch live soccer matches | General sports news, scores, or team-information apps without live match viewing |
| Horoscope/tarot/numerology guidance | Apps offering horoscope, tarot, numerology, or prediction-based guidance | Generic lifestyle or entertainment apps without these guidance functions |
| Watch face app | Apps that provide watch faces | General smartwatch utilities without watch-face functionality |
| Short drama | Apps that offer short drama content | Short story apps or regular long-form drama apps |
| Offline maps supporting cars, motorcycles, bicycles, etc. | Apps that provide offline maps, regardless of region, as long as they support at least one of the mentioned transportation modes | Map or navigation apps that require online access |
| Backing tracks for music practice | Apps that provide backing tracks or accompaniment for instrument practice | General music lessons without backing tracks |

