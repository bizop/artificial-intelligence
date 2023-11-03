import { OpenAILibrary } from './OpenAILibrary';

class MeetingMinutesGenerator {
  constructor(apiKey, audioFilePath) {
    this.apiKey = apiKey;
    this.audioFilePath = audioFilePath;
    this.openAILib = new OpenAILibrary(apiKey);
  }

  /**
   * Transcribes a meeting's audio and generates a summary of the minutes.
   */
  async generateMeetingMinutes() {
    try {
      // Transcribe the audio file to text
      console.log('Transcribing the audio...');
      const transcription = await this.openAILib.generateTranscription(this.audioFilePath);
      const transcribedText = transcription.results[0].alternatives[0].transcript;

      // Generate a summary of the transcription
      console.log('Generating meeting minutes...');
      const summary = await this.openAILib.generateTextSummary(transcribedText);

      // Output the meeting minutes
      console.log('Meeting minutes:');
      console.log(summary.response);

      return summary.response;
    } catch (error) {
      console.error('An error occurred while generating meeting minutes:', error);
      throw error;
    }
  }
}

// Example Usage:
// (Assuming the 'audioTest.mp3' is the recorded meeting audio file)

// Create an instance of the MeetingMinutesGenerator class
// const apiKey = 'Your-OpenAI-API-Key';
// const audioFilePath = 'audioTest.mp3';
// const mmGenerator = new MeetingMinutesGenerator(apiKey, audioFilePath);

// Generate and log the meeting minutes
// mmGenerator.generateMeetingMinutes().then((minutes) => console.log(minutes));
