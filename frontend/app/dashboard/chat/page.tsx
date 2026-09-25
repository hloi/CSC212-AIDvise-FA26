/*=============================================================================
CSC 212 — AI Academic Advising Platform
Copyright (c) 2026 Quinsigamond Community College — CSC 212
All rights reserved.
Author:   Sean Collins
GitHub:   https://github.com/LSilver17/CSC212---AI-Agent
=============================================================================*/
"use client"

import "@copilotkit/react-ui/styles.css";
import { CopilotChat } from "@copilotkit/react-ui";
import { Flex } from "@radix-ui/themes";
import { useEffect } from "react";

// Lib
import { useUserData } from "@/app/lib/account/user_context";
import type { UserData, UserMetadata, UserInterests, StudentData } from "@/app/lib/account/account_db_utils";
import type { AccountType } from "@/app/lib/account/account_type";

// Hooks
import { useCoAgent } from "@copilotkit/react-core";
import { authSession } from "@/app/lib/account/authSession";

/**
 * Customizes user welcome message based on account type and user data.
 * @param name 
 * @param accountType 
 * @param interests 
 * @returns 
 */
function choose_init_message(name: string, accountType: AccountType, interests: UserInterests | null = null): string {
  if(accountType === "Advisor") {
    var message_addition = "How can I help you today?";
    const message = `Hi, ${name}! ${message_addition}`;
    return message;
  } else if (accountType === "Student") {
    var message_addition = "How can I help you today?";
    if (interests && interests.Interests.length == 0) message_addition = "Tell me about your academic and extracurricular interests.";
    const message = `Hi, ${name}! ${message_addition}`;
    return message;
  }
  return "Hello!";
}

/**
 * Displays CopilotKit's CopilotChat{@link https://docs.copilotkit.ai/reference/v1/components/chat/CopilotChat} component with a customized user welcome message.
 */
export default function Chat() {
  const { userData, userMetadata, userInterests } : {userData: UserData, userMetadata: UserMetadata, userInterests: UserInterests} = useUserData();

  const accountType: AccountType = userMetadata?.AccountType;

  const { state, setState } = useCoAgent({
  name: "default",
  initialState: {
    user_id: userMetadata?.AccountID ?? null,
    account_type: accountType ?? null,
  },
});

useEffect(() => {
  const userId = userMetadata?.AccountID;

  if (!userId || !accountType) {
    return;
  }

  if (
    state?.user_id === userId &&
    state?.account_type === accountType
  ) {
    return;
  }

  console.log("Updating AIDvise agent identity:", {
    user_id: userId,
    account_type: accountType,
  });

  setState({
    ...state,
    user_id: userId,
    account_type: accountType,
  });
}, [
  userMetadata?.AccountID,
  accountType,
  state?.user_id,
  state?.account_type,
  setState,
]);

  // Set name and message
  var name: string = "User";
  if(accountType) name = accountType as string;
  if(userData && userData.Name.data) name = userData.Name.data;
  const message = choose_init_message(name, accountType, userInterests);

  return (
    <Flex width="100%" height="100%" direction="row">
      <CopilotChat
        labels={{
          title:"Advise Bot",
          initial: message,
        }}
        className="w-full h-full"
      />
    </Flex>
  );
}
