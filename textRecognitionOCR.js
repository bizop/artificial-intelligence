// *********************************************
// CODE DEVELOPED BY: NATHAN WILBANKS
// FIND ME ON LINKEDIN: https://www.linkedin.com/in/nathanwilbanks/
// FIND ME ON TWITTER: https://twitter.com/NathanWilbanks_
// LICENSE: MIT License
// VERSION: 11.12.2023
// COPYRIGHT: 2023 Nathan Wilbanks

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// *********************************************

import fs from 'fs';
import path from 'path';
import axios from 'axios';
import Tesseract from 'tesseract.js';

// Function to tell me what this is exactly via AI:
async function detectWithOpenAIFromText(text) {
  const prompt = `INSTRUCTIONS:
  This is text retrieved from a screenshot.
  What is this screenshot likely of? You can only choose one thing.
  Don't explain, just think about it and tell me exactly what it is along with your confidence score (e.g. "Mumbo Jumbo, 69% Confidence").
  If you can tell it's in a category for sure, determine what it is ("Amazon Product Review", not just "Product Review" | "Facebook", not just "Social Media Platform").
  Return only the object name and confidence score, NOTHING else (e.g. "Twitter Profile, 80% Confidence", NOT "This screenshot is likely of a "Twitter Profile", 80% Confidence.").

  SCREENSHOT TEXT:
  ${text}`;

  const messages = [
    {
      role: 'system',
      content: ``,
    },
    {
      role: 'user',
      content: prompt,
    },
  ];

  const response = await axios({
    method: 'post',
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiAPI}`,
    },
    data: {
      model: 'gpt-3.5-turbo-1106',
      messages: messages,
      max_tokens: 60,
      temperature: 1,
    },
  });

  const finalCompleteText = response.data.choices[0].message.content;
  return finalCompleteText.trim();
}

// Function to perform OCR on an image
async function performOCR(imagePath) {
  const {
    data: { text },
  } = await Tesseract.recognize(imagePath, 'eng');
  return text;
}

// Main function to process images using OCR
async function processImages() {
  const inputFolder = './images'; // Folder with images

  // Check if the directory exists
  if (!fs.existsSync(inputFolder)) {
    console.error(`The directory ${inputFolder} does not exist.`);
    return;
  }

  const files = fs.readdirSync(inputFolder);

  for (const file of files) {
    const filePath = path.join(inputFolder, file);

    // Check if the file is an image
    if (path.extname(filePath).match(/\.(jpg|jpeg|png|bmp|tiff)$/i)) {
      console.log('------------------------------------------------');
      console.log(`Processing: ${file}`);

      try {
        const text = await performOCR(filePath);

        console.log(`Text extracted from ${file}:`);
        // console.log(text);
        const whatItIs = await detectWithOpenAIFromText(text);
        // console.log('-------------------');
        console.log(whatItIs);
      } catch (error) {
        console.error(`Error processing ${file}: ${error.message}`);
      }
    } else {
      console.log(`Skipping non-image file: ${file}`);
    }
  }
}

processImages().then(() => {
  console.log('------------------------------------------------');
  console.log('Image processing complete.');
});
