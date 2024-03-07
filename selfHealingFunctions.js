import Anthropic from '@anthropic-ai/sdk';
import { exec } from 'child_process';
import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { setTimeout } from 'timers/promises';

const anthropic = new Anthropic({
  apiKey: process.env.CLAUDE_API_KEY,
});

function cleanCodeSnippet(snippet) {
  const startMarker = '```javascript';
  const endMarker = '```';
  if (snippet.startsWith(startMarker) && snippet.endsWith(endMarker)) {
    return snippet.slice(startMarker.length, -endMarker.length).trim();
  } else {
    return snippet;
  }
}

async function generateCode(userInput, retryCount = 5, initialDelay = 500) {
  let attempt = 0;
  let delay = initialDelay;
  while (attempt < retryCount) {
    try {
      const response = await anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 4000,
        temperature: 0.5,
        system: `You are modeling the mind of a the worlds greatest Javascript function code generator.
          IMPORTANT: Respond only with code, in its entire and complete form, starting with "\`\`\`javascript".
          IMPORTANT: DO NOT explain yourself or include examples as your code output is going to be ran directly in a VM and tested by another source.
          Generate the code based on the following user input:`,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Generate code based on the following user input:\n${userInput}`,
              },
            ],
          },
        ],
      });
      return cleanCodeSnippet(response.content[0].text.trim());
    } catch (error) {
      console.error(`Attempt ${attempt + 1} failed. Error: ${error.message}`);
      if (attempt === retryCount - 1) {
        throw new Error('All retry attempts failed');
      }
      await setTimeout(delay);
      delay *= 2;
      attempt++;
    }
  }
}

async function generateTests(code, retryCount = 5, initialDelay = 500) {
  let attempt = 0;
  let delay = initialDelay;
  while (attempt < retryCount) {
    try {
      const response = await anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 4000,
        temperature: 0.5,
        system: `You are modeling the mind of a the worlds greatest Javascript jest code generator.
          IMPORTANT: Respond only with code, in its entire and complete form, starting with "\`\`\`javascript".
          IMPORTANT: DO NOT explain yourself or include examples as your code output is going to be ran directly in a VM and tested by another source.
          IMPORTANT: DO NOT "import" or "require" the code or functions to test, that will be handled separately.
          Generate only the describe tests based on the following provided code:`,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Generate jest tests for the following code:\n${code}`,
              },
            ],
          },
        ],
      });
      return cleanCodeSnippet(response.content[0].text.trim());
    } catch (error) {
      console.error(`Attempt ${attempt + 1} failed. Error: ${error.message}`);
      if (attempt === retryCount - 1) {
        throw new Error('All retry attempts failed');
      }
      await setTimeout(delay);
      delay *= 2;
      attempt++;
    }
  }
}

async function regenerateCode(userInput, generatedCode, generatedTests, error, retryCount = 5, initialDelay = 500) {
  let attempt = 0;
  let delay = initialDelay;
  while (attempt < retryCount) {
    try {
      const response = await anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 4000,
        temperature: 0.5,
        system: `You are modeling the mind of a the worlds greatest Javascript function code generator.
          Think internally about the following code and failed tests, and return only the modified code adjusted to pass the tests that failed.
          IMPORTANT: Respond only with final code, in its entire and complete form, starting with "\`\`\`javascript".
          IMPORTANT: DO NOT explain yourself or include examples as your code output is going to be ran directly in a VM and tested by another source.
          
          {PREVIOUS FAILED CODE & TESTS OUTPUT}`,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: `[INSTRUCTIONS]:
                Regenerate code based on the following failed code and tests:
                
                [USER QUERY]:
                ${userInput}

                [FAILED CODE]:
                ${generatedCode}
                
                [ALL TESTS]:
                ${generatedTests}
                
                [FAILED TESTS]:
                ${error}
                
                [REVISED CODE]:
                \`\`\`javascript
                `,
              },
            ],
          },
        ],
      });
      return cleanCodeSnippet(response.content[0].text.trim());
    } catch (error) {
      console.error(`Attempt ${attempt + 1} failed. Error: ${error.message}`);
      if (attempt === retryCount - 1) {
        throw new Error('All retry attempts failed');
      }
      await setTimeout(delay);
      delay *= 2;
      attempt++;
    }
  }
}

async function runCodeInVM(code) {
  const useImport = code.includes('import');
  const fileExtension = useImport ? '.js' : '.cjs';
  const tempFile = `temp${fileExtension}`;

  // Write the code to a temporary file
  writeFileSync(tempFile, code);

  // Run the code in a virtual machine using child_process
  return new Promise((resolve, reject) => {
    exec(`node ${tempFile}`, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout);
      }
    });
  });
}

function extractDependencies(code) {
  const requireRegex = /require\(['"]([^'"]+)['"]\)/g;
  const importRegex = /import\s+(?:.+\s+from\s+)?['"]([^'"]+)['"]/g;
  const dependencies = new Set();
  let match;
  while ((match = requireRegex.exec(code)) !== null) {
    dependencies.add(match[1]);
  }
  while ((match = importRegex.exec(code)) !== null) {
    dependencies.add(match[1]);
  }
  return Array.from(dependencies);
}

async function installDependencies(code) {
  // Extract the dependencies from the code (e.g., using regular expressions)
  const dependencies = extractDependencies(code);
  // Install the dependencies using npm
  for (const dependency of dependencies) {
    await new Promise((resolve, reject) => {
      exec(`npm install ${dependency}`, (error, stdout, stderr) => {
        if (error) {
          reject(error);
        } else {
          resolve();
        }
      });
    });
  }
}

async function runTests(code, tests) {
  const tempFile = `temp.test.js`;

  // Write the code and tests to a single temporary file
  writeFileSync(tempFile, `${code}\n${tests}`);

  // Run the tests using the local installation of Jest
  return new Promise((resolve, reject) => {
    exec(`npx jest ${tempFile}`, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout);
      }
    });
  });
}

async function outputFinalCode(code, tests) {
  const outputDir = 'output';
  const useImport = code.includes('import');
  const fileExtension = useImport ? '.js' : '.cjs';
  const codeFile = `code${fileExtension}`;
  const testFile = `tests.js`;

  // Check if the output directory exists
  if (!existsSync(outputDir)) {
    // Create the output directory if it doesn't exist
    mkdirSync(outputDir);
  }

  // Write the final code to a file in the output directory
  writeFileSync(`${outputDir}/${codeFile}`, code);
  writeFileSync(`${outputDir}/${testFile}`, tests);
}

async function generateDocumentation(code) {
  const prompt = `You are modeling the mind of a the worlds greatest code documentation generator.
  
    Please generate a README.md file for the following code, including overviews, explainations, code examples, input params, expected outputs, etc.

    \`\`\`javascript
    ${code}
    \`\`\`

    Include sections for:
    - Overview
    - Installation
    - Usage
    - API Documentation
    - License

    Respond only with the final readme documentation in its entire and complete form, starting with the readme title.
    `;

  const response = await anthropic.messages.create({
    model: 'claude-3-opus-20240229',
    max_tokens: 4000,
    temperature: 0.5,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  });

  const readme = response.content[0].text.trim();

  const readmePath = join(process.cwd(), `./output/README.md`);
  writeFileSync(readmePath, readme);

  console.log('README.md generated successfully!');
}

async function main(userInput) {
  try {
    let generatedCode = await generateCode(userInput);
    let generatedTests = await generateTests(generatedCode);

    while (true) {
      console.log('***CODE***');
      console.log(generatedCode);
      console.log('\n');
      console.log('***TESTS***');
      console.log(generatedTests);
      console.log('\n');
      console.log('***********');
      console.log('\n');

      // Install dependencies
      await installDependencies(generatedCode);

      // Run the code in a virtual machine & do tests
      try {
        const codeOutput = await runCodeInVM(generatedCode);
        console.log('Code output:', codeOutput);
        await runTests(generatedCode, generatedTests);
        console.log('Code Passed All Tests successfully!');
        console.log('Generating documentation...');
        await generateDocumentation(generatedCode);
        break; // Exit the loop if tests pass
      } catch (error) {
        console.error('Test failed:', error);
        generatedCode = await regenerateCode(userInput, generatedCode, generatedTests, error);
      }
    }

    // Output the final working code
    await outputFinalCode(generatedCode, generatedTests);
    console.log('Code generation completed successfully!');
  } catch (error) {
    console.error('An error occurred:', error);
  }
}

main('fibonacci sequence generator');
