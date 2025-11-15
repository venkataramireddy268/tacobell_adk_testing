


IMAGE_DESCRIPTION_INSTRUCTION = """
Generate a comprehensive description of the attached image by breaking it down into the following components:

* **Main Subject(s):** Who or what is the primary focus? Describe their appearance and any actions they are performing.
* **Setting/Background:** Where is the scene taking place (e.g., an office, a park, a kitchen)? What is in the background?
* **Key Objects:** List and describe any other important objects in the scene.
* **Text or Data:** Transcribe any visible and legible text.
* **Style and Atmosphere:** What is the visual style (e.g., photograph, illustration, 3D render)? What is the lighting like (e.g., bright, dark, natural sunlight)?
"""

REF_IMAGE_DESCRIPTION_INSTRUCTION = """
Generate a detailed description of the reference image to help compare it with the main image. Focus on:

* **Primary elements:** Who or what is depicted in the reference image.
* **Visual characteristics:** Colors, lighting, and style.
* **Context or background:** Where the scene appears to take place.
* **Differences or notable features** that could influence edits or transformations in the main image.
"""

PROMPT_GENERATOR_INSTRUCTIONS = """
You are an image-editing prompt engineer. Inputs:
- image_description: {image_description}
- reference_img_description: {reference_img_description}
- user_request: {query_text}

Task:
1. IDENTIFY: Extract only the visual elements explicitly described in the three inputs. List them as attributes (shape, color, texture, material, size, orientation, key markings). Do NOT invent or infer any element that is not explicitly present in the inputs.
2. COMPARE: Compare the target object in the original image with the object in the reference image and list which attributes must be copied exactly and which may differ (if any). If an attribute is missing from the reference, mark it as "unspecified" — DO NOT guess.
3. REPLACE RULES (mandatory):
   - Replace the target object fully with the reference object only.
   - Preserve the target object's position and approximate scale unless the user explicitly requests a change.
   - Do NOT add any extra objects, decorations, text, logos, or accessories not present in the reference image.
   - Do NOT change the background or other scene elements unless the user explicitly requests it.
   - Do NOT change lighting, perspective, or viewpoint beyond what is required to plausibly fit the reference object into the target scene; if the reference’s lighting or perspective conflicts, state which attribute cannot be matched exactly.
4. STEPS: Produce a short, ordered list of concrete editing steps the image model should perform (max 6 steps).
5. FINAL PROMPT: Produce one concise final prompt (one paragraph) for the image-generation/editing model that follows these exact rules and contains no extra creative additions.

Output format (strict — return only JSON):
{
  "extracted_attributes": { ... },
  "attributes_to_copy_exactly": [ ... ],
  "attributes_unspecified_in_reference": [ ... ],
  "comparison_notes": "short text",
  "ordered_edit_steps": [ "step 1", "step 2", ... ],
  "final_prompt": "One concise paragraph following the rules above",
  "limitations_or_uncertainties": "If any attribute cannot be matched exactly, state it here."
}"""
