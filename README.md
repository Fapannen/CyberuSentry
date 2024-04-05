<img src="https://github.com/Fapannen/CyberuSentry/blob/main/img/v1.png" width="800" height="450" />

# CyberuSentry
CyberuSentry is a mixture of 4 words which describe the nature of this project.
  - [Cyber](https://dictionary.cambridge.org/dictionary/english/cyber) - Involving, using, or relating to computers 
  - [Cerberus](https://en.wikipedia.org/wiki/Cerberus) - A three-headed guardian dog from the Greek mythology
  - [Sentry](https://dictionary.cambridge.org/dictionary/english/sentry) - A soldier who guards a place, usually by standing at its entrance
  - [Entry](https://dictionary.cambridge.org/dictionary/english/entry) - The act of entering a place

CyberuSentry is a system that detects customer frauds in accomodation booking. When renting a place to stay, such as via AirBnb or Booking.com, the customers sometimes declare that X people will be staying
at the place, only to find out that there were actually Y people ( Y > X ) staying at the place. This results in financial loss for the renter, though it is difficult to prove that there were more people than
declared. CyberuSentry should serve as a tool to assess the amount of unique people that have visited the accomodation and help the renters provide a proof that the accomodation was used by more people than
originally declared.

CyberuSentry uses a neural network to determine how many people visited the accomodation during a specific time frame. Note that the system **does not perform Face Recognition**, it only performs **Face Detection**. 
The main difference is that unlike Face Recognition, Face Detection does not connect the face to any personal data and thus the identity of the person is not known and is completely irrelevant. As such, the system
should be safe to use in accomodation renting, of course by placing the surveillance camera to a neutral place where it does not invade personal privacy. The guests should also be informed about the fact that there
is such a system in place to avoid potential disputes.

# Sources
- Images were generated using [Gencraft](https://gencraft.com/) website. All images used were created under a paid subscription.
- [Cambridge Dictionary](https://dictionary.cambridge.org/)
