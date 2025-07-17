README - Behavioral Data Description
This README describes the columns and data contained in the behavioral dataset files (.csv or .xlsx).

Gabor_rotation : indicates the relative rotation direction of the second Gabor grating compared to the first one: "clockwise" or "counterclockwise".

Gabor_Direction: Represents the absolute orientation of the first Gabor grating:
- "left": oriented counterclockwise relative to vertical.
- "right": oriented clockwise relative to vertical.

Time_Action_Choice: Timestamp  of the participant's self-initiated action (in Active condition only), corresponding to a button press:

Rt_choice: Reaction time (in seconds) to perform the action choice, measured from the onset of the cue (fixation dot) in Active trials.

Agency_Condition: Indicates the experimental condition:
- "A" = Active (self-generated and predicted first grating)
- "P" = Passive (externally-generated and unpredicted first grating)

Participant_Response: Response to the discrimination task (second grating's rotation):
- "num_9" = participant pressed key 9 indicating clockwise rotation.
- "a" = participant pressed key A indicating counterclockwise rotation.

Time_Response: Timestamp of the participant's discrimination response.

Reaction_Time: Reaction time (in seconds) for the discrimination response, measured from the second grating disappearance.

Discrimination: Present only in "Go" trials (relevant condition) : 
- Value 1 if the participant gave the correct response (clockwise or counterclockwise) or 0 for incorrect response.
- Missing/blank in "No-go" trials (irrelevant condition).

Good_response: Evaluates overall correctness across both discrimination and Go/No-go tasks:
- 1 = correct behavior: participant responded correctly in Go trials and correctly withheld a response in in No-go trials.
- 0 = incorrect behavior: Responded when they should not have, or failed to respond when required, or gave the wrong discrimination response.


For questions or further clarification regarding the dataset, please contact the corresponding author of the associated publication.
thomas.holstein.pro@proton.me