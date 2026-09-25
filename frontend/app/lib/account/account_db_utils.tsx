/*=============================================================================
CSC 212 — AI Academic Advising Platform
Copyright (c) 2026 Quinsigamond Community College — CSC 212
All rights reserved.
Author:   Sean Collins
Co-Author: Luca Silver
GitHub:   https://github.com/LSilver17/CSC212---AI-Agent
Description: 
    A set of utility functions and types for connecting user accounts with
    the database.
=============================================================================*/
"use server";

import sqlite3 from "sqlite3";
import { open } from "sqlite";
import bcrypt from "bcrypt";
import { redirect } from "next/navigation";
import path from "path";
// import data from "../../../../config.json";
import fs from "fs";
const configPath = path.resolve(process.cwd(), "../config.json");

const data = JSON.parse(
    fs.readFileSync(configPath, "utf8")
);
// User
import type { AccountType } from "@/app/lib/account/account_type";
import type { Alert, EventAlert, ClassAlert } from "@/app/lib/alerts/alert";
import { createEventAlert, createClassAlert } from "@/app/lib/alerts/alert";
import { authSession } from "@/app/lib/account/authSession";
import { UserField } from "@/app/lib/account/user_fields";
import { ensureFieldFormat } from "@/app/lib/form/user_fields_format_test";
import { UserContextProvider } from "./user_context";

/**
 * Returns path to database configured in database_config.json.
 * @returns Database path.
 */
function dbPath() {
  const dbPath = path.join(
    process.cwd(),
    "..",
    `${data.database_config.db_name}.db`,
  );
  return dbPath;
}

/**
 * Creates database object for database configured in database_config.json.
 * @param path - Path to the database, by default the configured DB path.
 * @returns SQLite database object.
 */
async function openDB(path = dbPath()) {
  const db = await open({
    filename: path,
    driver: sqlite3.Database,
  });
  await db.exec("PRAGMA foreign_keys = ON");
  return db;
}

/**
 * Literal type for holding login results, storing credentials or error logs
 * depending on success.
 */
export type LoginResult =
  | { success: true; username: string; account_type: AccountType; id: string }
  | { success: false; error: string };

/**
 * Literal type for holding login results, storing credentials or error logs
 * depending on success.
 */
export type CreationResult =
  | {
      success: true;
      id: string;
      username: string;
      account_type: AccountType;
      academic_id: string;
    }
  | { success: false; error: string };

/**
 * Validates user-provided credentials against database entries. On successful login, returns on object with the validated credentials, returning an object with an error message otherwise.
 * @param {string} username - The provided username credential.
 * @param {string} password - The provided password credential, hashed using the same algorithm used on insertion and compared with the hashed value in the database.
 * @returns {Promise<LoginResult>} A promise containing an object with login results.
 * @authors Sean Collins, Luca Silver
 */
export async function validate_credentials(
  username: string,
  password: string,
): Promise<LoginResult> {
  const normalizedUsername = username.trim().toLowerCase();
  let db;

  try {
    db = await openDB(dbPath());

    const credential = await db.get(
      `SELECT Username, Password, ID, AccountType
             FROM Users
             WHERE lower(trim(Username)) = ?`,
        normalizedUsername,
    );

    // Check before accessing credential.Password
    if (!credential) {
      return {
        success: false,
        error: "User not found",
      };
    }

    const valid = await bcrypt.compare(password, credential.Password);

    if (!valid) {
      return {
        success: false,
        error: "Incorrect password",
      };
    }

    return {
      success: true,
      username: credential.Username,
      account_type: credential.AccountType,
      id: credential.ID,
    };
  } catch (error) {
    console.error("Credential validation error:", error);

    return {
      success: false,
      error: "Login database error",
    };
  } finally {
    if (db) {
      await db.close();
    }
  }
}

/**
 * Deletes any invalid entries created during the user creation process if an error is thrown.
 * @param db - SQLite database object.
 * @param username - Username field of entry to delete.
 */
async function delete_invalid_entry(db: any, username: string) {
  if (db) {
    const user = await db.get(
      "SELECT Username, AccountType, ID FROM Users WHERE Username = ?",
      username,
    );
    if (user) {
      await db.run("DELETE FROM Users WHERE Username = ?", username);
    }
  }
}

/**
 * Validation result for the id_validation() method.
 */
type ValidationResult =
  | { valid: true; table: string }
  | { valid: false; err: string };

/**
 * Ensures that the provided student/advisor ID has an equivalent within the database before linking an account to it.
 * @param db - SQLite database object.
 * @param account_type - Student / Advisor
 * @param person_id - Person's academic student/advisor ID stored in the database.
 * @returns {Promise<ValidationResult>} Promise with an object containing table name if ID is valid and an error message if not.
 */
async function id_validation(
  db: any,
  account_type: AccountType,
  person_id: string,
): Promise<ValidationResult> {
  if (account_type) {
    const table = account_type === "Student" ? "Students" : "Advisors";
    const personID = await db.get(
      `SELECT ID FROM ${table} WHERE ID = ?`,
      person_id,
    );
    if (!personID) {
      const result: ValidationResult = {
        valid: false,
        err: `${account_type} does not exist`,
      };
      return result;
    }
    const associatedAcc = await db.get(
      `SELECT ParentID FROM ${table} WHERE ID = ?`,
      person_id,
    );
    if (associatedAcc.ParentID != null) {
      const result: ValidationResult = {
        valid: false,
        err: "There is already an account associated with this ID",
      };
      return result;
    }
    const result: ValidationResult = {
      valid: true,
      table: table,
    };
    return result;
  } else {
    throw "Invalid account type";
  }
}

/**
 * Creates a new User entry in the database. Accounts can only be created for existing Student or Advisor
 * entries in the database. Will return unsuccessful if the username is taken, if there is no person with
 * an equivalent academic ID in the database, or if there is already an account associated with that person.
 * @param username - Username of the new account.
 * @param password - User's password, to be salted and hashed before entry.
 * @param account_type - Type of account. Determines privileges and which table to check.
 * @param person_id - Academic ID of the person to be found in the database (In the Students/Advisors tables).
 * @returns {Promise<CreationResult>} An object detailing the result of the creation attempt.
 * @authors Sean Collins, Luca Silver
 */
export async function create_user(
  username: string,
  password: string,
  account_type: AccountType,
  person_id: string,
): Promise<CreationResult> {
  var db;
  const validAccountTypes = ["Student", "Advisor"];
  const normalizedUsername = username.trim().toLowerCase();
  try {
    if (!account_type || !validAccountTypes.includes(account_type)) {
      throw "Invalid account type";
    }

    db = await openDB(dbPath());

    const existingUser = await db.get(
      `SELECT Username
     FROM Users
     WHERE lower(trim(Username)) = ?`,
      normalizedUsername,
    );

    
    if (existingUser) {
      const result: CreationResult = {
        success: false,
        error: "User already exists",
      };
      return result;
    }

    // Ensures ID matches existing student/advisor entry and there is no account already associated
    const valid = await id_validation(db, account_type, person_id);
    let tableName = null;
    if (valid.valid) {
      tableName = valid.table;
    } else {
      throw valid.err;
    }

    // Hash the password before storing it in the database
    const saltRounds = 10;
    const salt = await bcrypt.genSalt(saltRounds);
    const hash = await bcrypt.hash(password, salt);

    await db.run(
      "INSERT INTO Users (Username, Password, AccountType) VALUES (?, ?, ?)",
      normalizedUsername,
      hash,
      account_type,
    );

    // Pull inserted credentials to return in the session
    const credential = await db.get(
      "SELECT Username, AccountType, ID FROM Users WHERE lower(trim(Username)) = ?",
      normalizedUsername
    );

    // Link existing Student/Advisor record to new User entry
    await db.run(
      `UPDATE ${tableName} SET ParentID = ? WHERE ID = ?`,
      credential.ID,
      person_id,
    );

    const result: CreationResult = {
      success: true,
      username: credential.Username,
      account_type: credential.AccountType,
      id: credential.ID,
      academic_id: person_id,
    };

    return result;
  } catch (e) {
    const result: CreationResult = {
      success: false,
      error: `Database error: ${e}`,
    };
    await delete_invalid_entry(db, username);
    return result;
  } finally {
    if (db) {
      await db.close();
    }
  }
}

/**
 * Contains success status of database operations performed by database handler functions like
 * {@link update_user_entry} and {@link delete_account}.
 */
export type Result =
  | { success: true; query?: string }
  | { success: false; error: string };

/**
 * @param type - Account type (Student / Advisor)
 * @returns Name of the corresponding table for provided account type.
 */
function accountTypeToTable(type: string | undefined) {
  var tableName;
  switch (type) {
    case "Student":
      tableName = "Students";
      break;
    case "Advisor":
      tableName = "Advisors";
      break;
    default:
      tableName = null;
      break;
  }
  return tableName;
}

/**
 * Updates user entry in database when account details are changed, validating session and ensuring user data exists beforehand.
 * @param userData - Object from UserContext containing information about the user.
 * @param userMetadata - Object from UserContext containing information about the user's account.
 * @param newVal - User-provided value to be inserted.
 * @param updatedField - Field of the new value.
 * @returns A promise with an object detailing the result.
 */
export async function update_user_entry(
  userData: UserData,
  userMetadata: UserMetadata,
  newVal: string | null,
  updatedField: string,
): Promise<Result> {
  var db;
  const session = await authSession();
  if (!session) {
    redirect("/login");
  }
  try {
    // Ensure data exists
    if (!userData) {
      throw Error("User data not found");
    } else if (!userMetadata) {
      throw Error("User metadata not found");
    } else if (!session) {
      throw Error("No active session");
    }

    db = await openDB(dbPath());

    // setup constants for query
    const table = accountTypeToTable(userMetadata.AccountType);
    const checkFormat = ensureFieldFormat(updatedField, newVal);
    if (!checkFormat.success) {
      throw Error(`${checkFormat.error}`);
    }

    // run query
    const query = `UPDATE ${table} SET ${updatedField} = ? WHERE ParentID = ?`;
    await db.run(query, newVal, session.user.id);

    const result: Result = {
      success: true,
      query: query,
    };
    return result;
  } catch (e: any) {
    const result: Result = {
      success: false,
      error: `${e.message}`,
    };
    return result;
  } finally {
    if (db) {
      await db.close();
    }
  }
}

/**
 * Deletes the account entry of the user with current active session. Student/Advisor info remains in the academic database.
 * Make sure to clear the session after this function is called using NextAuth's signOut() hook.
 * @returns A promise with an object detailing the result.
 */
export async function delete_account(): Promise<Result> {
  let db;
  const session = await authSession();
  if (!session) {
    redirect("/login");
  }
  try {
    if (!session?.user?.id) {
      throw Error("Unauthorized session");
    }

    db = await openDB(dbPath());
    const userID = session.user.id;
    const account_type = session.user.account_type;
    const account_table = account_type === "Student" ? "Students" : "Advisors";

    if (!account_table) {
      throw Error("No valid account type");
    }

    await db.run(
      `UPDATE ${account_table} SET ParentID = ? WHERE ParentID = ?`,
      null,
      userID,
    );
    await db.run("DELETE FROM Users WHERE ID = ?", userID);
    const result: Result = {
      success: true,
    };
    return result;
  } catch (e: any) {
    const result: Result = {
      success: false,
      error: `${e.message}`,
    };
    return result;
  } finally {
    if (db) {
      await db.close();
    }
  }
}

export type StudentData = {
  Name: UserField;
  GPA: UserField;
  CreditsEarned: UserField;
  IntendedGraduationTerm: UserField;
  AdvisorID: UserField;
  StudentID: UserField;
};
export type AdvisorData = {
  Name: UserField;
  AdvisorID: UserField;
};

/**
 * User data object for advisor/student specific data, used in context creation. Each property
 * is stored as a UserField. Refer to {@link UserField} documentation to see how user fields
 * are stored.
 *
 * @property Name (student)
 * @property GPA (student)
 * @property CreditsEarned (student)
 * @property IntendedGraduationTerm (student)
 * @property AdvisorID (student)
 * @property StudentID (student)
 * @property Name (advisor)
 * @property AdvisorID (advisor)
 */
export type UserData = StudentData | AdvisorData;

/**
 * Object storing account metadata like type, username, and ID.
 * @property AccountType: {@link AccountType}
 * @property Username: string
 * @property AccountID: string
 */
export type UserMetadata = {
  AccountType: AccountType;
  Username: string;
  AccountID: string;
};

/** Object storing seen and unseen alerts.
 * @property UnseenAlerts: {@link Alert}[]
 * @property SeenAlerts: {@link Alert}[]
 */
export type UserAlerts = {
  UnseenAlerts: Alert[];
  SeenAlerts: Alert[];
};

/**
 * Object of user interests tracked by the agent.
 */
export type UserInterests = {
  Interests: string[];
};

/**
 * Object containing information about a student under an advisor.
 */
export type Student = {
  ID?: string;
  Name: string | null;
  GPA: number | null;
  CreditsEarned: number | null;
  IntendedGraduationTerm: string | null;
};

/**
 * List of advisor's students.
 */
export type UserStudents = {
  Students: Student[];
};

/**
 * Result object tracking success of data fetch.
 */
export type UserDataResult =
  | {
      data: UserData;
      success: true;
    }
  | {
      success: false;
      error: string;
    };
/**
 * Grabs user data from database to fill out context for Advisor/Student.
 * @returns An object containing data or an error.
 */
export async function grabUserData(): Promise<UserDataResult> {
  const session = await authSession();
  if (!session) {
    redirect("/login");
  }
  const id = session?.user?.id;
  var db;
  try {
    db = await openDB(dbPath());

    const accountType = await db.get(
      "SELECT AccountType FROM Users WHERE ID = ?",
      id,
    );

    if (accountType.AccountType === "Student") {
      const userInfo = await db.get(
        "SELECT Name, GPA, CreditsEarned, IntendedGraduationTerm, AdvisorID, ID FROM Students WHERE ParentID = ?",
        id,
      );
      const advisor_name = await db.get(
        "SELECT Name FROM Advisors WHERE ID = ?",
        userInfo.AdvisorID,
      );
      const advisor_string = advisor_name?.Name ?? "Advising Center";
      const data: UserData = {
        Name: {
          title: "Name",
          data: userInfo.Name,
          editable: false,
        },
        GPA: {
          title: "GPA",
          data: userInfo.GPA,
          editable: false,
        },
        CreditsEarned: {
          title: "Credits Earned",
          data: userInfo.CreditsEarned,
          editable: false,
        },
        IntendedGraduationTerm: {
          title: "Expected Graduation",
          data: userInfo.IntendedGraduationTerm,
          editable: false,
        },
        AdvisorID: {
          title: "Advisor",
          data: advisor_string,
          editable: false,
        },
        StudentID: {
          title: "Student ID",
          data: userInfo.ID,
          editable: false,
        },
      };
      const fieldData: UserDataResult = {
        data: data,
        success: true,
      };
      return fieldData;
    } else if (accountType.AccountType === "Advisor") {
      const userInfo = await db.get(
        "SELECT Name, ID FROM Advisors WHERE ParentID = ?",
        id,
      );
      const data: UserData = {
        Name: {
          title: "Name",
          data: userInfo.Name,
          editable: true,
        },
        AdvisorID: {
          title: "Advisor ID",
          data: userInfo.ID,
          editable: false,
        },
      };
      const fieldData: UserDataResult = {
        data: data,
        success: true,
      };
      return fieldData;
    } else {
      throw `Account of type ${accountType.AccountType} with ID ${id} does not exist`;
    }
  } catch (e) {
    const error: UserDataResult = {
      success: false,
      error: `ERROR: ${e}`,
    };
    return error;
  } finally {
    if (db) {
      await db.close();
    }
  }
}

/**
 * Handles errors from grabUserData(), returning an object with the data or redirecting to login if user is nonexistent.
 * @returns UserData object.
 */
export async function getUserObject() {
  // grab user data from database
  try {
    const userData = await grabUserData();
    if (!userData?.success) {
      if ("error" in userData) {
        throw new Error(userData.error);
      } else {
        throw new Error("Undefined error");
      }
    } else if (userData.success && "data" in userData) {
      const userObject = userData.data;
      return userObject;
    } else {
      throw new Error("No data found");
    }
  } catch (e: any) {
    const error = e.message;
    console.log(`ERROR: ${error}`);
    redirect("/login");
  }
}

/**
 * Base user context for general account fields.
 * @property userData: {@link UserData}
 * @property userMetadata: {@link UserMetadata}
 */
export interface Context {
  userData: UserData;
  userMetadata: UserMetadata;
}
/**
 * Extended {@link Context} for student accounts.
 * @property userAlerts: {@link UserAlerts}
 * @property userInterests: {@link UserInterests}
 */
export interface StudentContext extends Context {
  userAlerts: UserAlerts;
  userInterests: UserInterests;
}
/**
 * Extended {@link Context} for advisor accounts.
 * @property userStudents: {@link UserStudents}
 */
export interface AdvisorContext extends Context {
  userStudents: UserStudents;
}

/**
 * Storage object for database alert fetches.
 * @property unseen: {@link Alert}[]
 * @property seen: {@link Alert}[]
 */
type AlertReturn = {
  unseen: Alert[];
  seen: Alert[];
};

/**
 * Gathers values for all UserContext state variables into an object (refer to {@link UserContextProvider})
 * @param account_type - Student / Advisor
 * @returns Object containing UserContext data, null if account type is invalid (should not occur).
 */
export async function get_curr_context(
  account_type: AccountType,
): Promise<StudentContext | AdvisorContext | null> {
  // session validation
  const session = await authSession();

  if (!session) {
    redirect("/login");
  }

  const userMetadata: UserMetadata = {
    AccountType: session.user.account_type,
    Username: session.user.username,
    AccountID: session.user.id,
  };

  if (account_type === "Student") {
    const userData = (await getUserObject()) as StudentData;
    const userAlerts = await get_alerts(userData.StudentID.data);
    const interests = await get_interests(userData.StudentID.data);

    const userInterests: UserInterests = {
      Interests: interests,
    };
    const currContext = {
      userData: userData,
      userMetadata: userMetadata,
      userAlerts: userAlerts,
      userInterests: userInterests,
    };
    return currContext;
  } else if (account_type === "Advisor") {
    const userData = (await getUserObject()) as AdvisorData;
    const userStudents: UserStudents = {
      Students: await student_fill(userData.AdvisorID.data),
    };
    const currContext = {
      userData: userData,
      userMetadata: userMetadata,
      userStudents: userStudents,
    };
    return currContext;
  }
  return null;
}

/**
 * Grabs student interests from database based on provided student ID.
 * @param student_id - Academic ID of the student.
 * @returns A list of interests.
 */
async function get_interests(student_id: string) {
  var db;
  var interests: string[] = [];
  try {
    db = await openDB(dbPath());
    const db_interests = await db.all(
      `SELECT Interest FROM Interests WHERE ParentID = ?`,
      student_id,
    );
    for (let interest of db_interests) {
      interests.push(interest.Interest);
    }
  } catch (e) {
    console.log(`ERROR: ${e}`);
    return interests;
  } finally {
    if (db) {
      await db.close();
    }
  }
  return interests;
}

/**
 * Returns an alerts object with seen and unseen alerts, to be stored in UserContext.
 * @param student_id - Academic student ID.
 * @returns Promise with object containing seen and unseen alerts.
 */
export async function get_alerts(student_id: string): Promise<UserAlerts> {
  const alerts = await alert_fill(student_id);
  const userAlerts: UserAlerts = {
    UnseenAlerts: alerts.unseen,
    SeenAlerts: alerts.seen,
  };
  return userAlerts;
}

/**
 * Sorts list of event alerts chronologically from furthest to closest.
 * @param alerts - Unsorted list of event alerts.
 * @returns Sorted list of event alerts.
 */
function sort_newest_oldest(alerts: Alert[]): Alert[] {
  return alerts.sort((a: Alert, b: Alert) => {
    const a_datetime: string = a.date + a.time;
    const b_datetime: string = b.date + b.time;
    // if b occurs after a, it appears first in list and vice versa
    if (a_datetime < b_datetime) {
      return 1;
    } else {
      return -1;
    }
  });
}

/**
 * Returns object with sorted seen and unseen alerts for events and courses. Events
 * are sorted with {@link sort_newest_oldest}.
 * @param student_id - Academic ID of student.
 * @returns Promise for object containing sorted event and course alerts.
 */
async function alert_fill(student_id: string): Promise<AlertReturn> {
  var db;
  let event_unseenAlerts: Alert[] = [];
  let event_seenAlerts: Alert[] = [];
  let course_unseenAlerts: Alert[] = [];
  let course_seenAlerts: Alert[] = [];
  try {
    db = await openDB(dbPath());
    const db_event_alerts = await db.all(
      `SELECT EventID, AlertStatus, ID FROM RelevantEvents WHERE ParentID = ?`,
      student_id,
    );
    for (const alert of db_event_alerts) {
      const eventID = alert.EventID;
      const dbEvent = await db.get(
        `SELECT Name, Description FROM Events WHERE ID = ?`,
        eventID,
      );
      const dbEventTimes = await db.get(
        `SELECT Date, StartTime, EndTime FROM EventDates WHERE ParentID = ?`,
        eventID,
      );
      const time = `${dbEventTimes.StartTime} - ${dbEventTimes.EndTime}`;
      const newAlert = createEventAlert(
        dbEvent.Name,
        dbEvent.Description,
        alert.AlertStatus,
        time,
        dbEventTimes.Date,
        alert.ID,
      );
      if (newAlert.status === "Unseen") {
        event_unseenAlerts.push(newAlert);
      } else if (newAlert.status === "Seen") {
        event_seenAlerts.push(newAlert);
      }
    }
    const db_course_alerts = await db.all(
      `SELECT AlertStatus, ChangeID, ID FROM StudentSectionStatusChanges WHERE ParentID = ?`,
      student_id,
    );
    for (const alert of db_course_alerts) {
      const alert_section_id = await db.get(
        `SELECT SectionID, ChangeTime, NewStatus FROM SectionStatusChanges WHERE ID = ?`,
        alert.ChangeID,
      );
      const section_info = await db.get(
        `SELECT SectionNum, Instructor, StartDate, EndDate, Status, MaxSeats, SeatsLeft, Method, Location, ParentID FROM Sections WHERE ID = ?`,
        alert_section_id.SectionID,
      );
      const course_id = await db.get(
        `SELECT ParentID FROM CoursesOffered WHERE ID = ?`,
        section_info.ParentID,
      );
      const class_info = await db.get(
        `SELECT Department, Code, Name, Description, Credits, Requirements, SemestersOffered FROM Courses WHERE ID = ?`,
        course_id.ParentID,
      );
      const [date, time] = alert_section_id.ChangeTime.split(" ");
      const meet_times = await db.all(
        `SELECT Day, StartTime, EndTime FROM MeetTimes WHERE ParentID = ?`,
        alert_section_id.SectionID,
      );
      let meet_schedule: string = "";
      for (const time of meet_times) {
        meet_schedule += `${time.Day}: ${time.StartTime}-${time.EndTime} `;
      }
      const newAlert = createClassAlert(
        `Section ${alert_section_id.NewStatus}`,
        alert.AlertStatus,
        date,
        time,
        class_info.Department,
        class_info.Code,
        class_info.Name,
        class_info.Description,
        class_info.Credits,
        class_info.Requirements,
        meet_schedule,
        class_info.SemestersOffered,
        section_info.SectionNum,
        section_info.Status,
        section_info.MaxSeats,
        section_info.SeatsLeft,
        section_info.Method,
        section_info.Location,
        alert.ID,
      );
      if (newAlert.status === "Unseen") {
        course_unseenAlerts.push(newAlert);
      } else if (newAlert.status === "Seen") {
        course_seenAlerts.push(newAlert);
      }
    }
  } catch (e) {
    console.log(`ERROR: ${e}`);
    const unseen = sort_newest_oldest([
      ...event_unseenAlerts,
      ...course_unseenAlerts,
    ]);
    const seen = sort_newest_oldest([
      ...event_seenAlerts,
      ...course_seenAlerts,
    ]);
    const alerts: AlertReturn = {
      unseen: unseen,
      seen: seen,
    };
    return alerts;
  } finally {
    if (db) {
      await db.close();
    }
  }
  const unseen = sort_newest_oldest([
    ...event_unseenAlerts,
    ...course_unseenAlerts,
  ]);
  const seen = sort_newest_oldest([...event_seenAlerts, ...course_seenAlerts]);
  const alerts: AlertReturn = {
    unseen: unseen,
    seen: seen,
  };
  return alerts;
}

/**
 * Changes event status in database from "Unseen" to "Seen" and then prepends unseen alerts to seen alerts list.
 * @param alerts - Current user alerts object.
 * @returns Updated user alerts object.
 */
export async function mark_alerts_as_seen(
  alerts: UserAlerts,
): Promise<UserAlerts> {
  var db;
  const unseen_alerts = alerts.UnseenAlerts;
  try {
    db = await openDB(dbPath());
    // Updates database entry
    for (const alert of unseen_alerts) {
      if (alert.type === "Event") {
        await db.run(
          `UPDATE RelevantEvents SET AlertStatus = ? WHERE ID = ?`,
          "Seen",
          alert.id,
        );
      } else if (alert.type === "Class") {
        await db.run(
          `UPDATE StudentSectionStatusChanges SET AlertStatus = ? WHERE ID = ?`,
          "Seen",
          alert.id,
        );
      }
    }
    const SeenAlerts = sort_newest_oldest([
      ...alerts.UnseenAlerts,
      ...alerts.SeenAlerts,
    ]);
    const userAlerts: UserAlerts = {
      UnseenAlerts: [],
      SeenAlerts: SeenAlerts,
    };
    return userAlerts;
  } catch (e) {
    console.log(`ERROR: ${e}`);
    return alerts;
  } finally {
    if (db) {
      await db.close();
    }
  }
}

/**
 * Grabs all students under an advisor's guidance and places them in a list to be stored in UserContext.
 * @param advisor_id - Advisor shared by the students.
 * @returns Promise with a list of all students.
 */
async function student_fill(advisor_id: string): Promise<Student[]> {
  var db;
  let students: Student[] = [];
  try {
    db = await openDB(dbPath());

    const db_students = await db.all(
      `SELECT ID, Name, GPA, CreditsEarned, IntendedGraduationTerm FROM Students WHERE AdvisorID = ?`,
      advisor_id,
    );
    for (let student of db_students) {
      const newStudent: Student = {
        Name: student.Name,
        ID: student.ID,
        GPA: student.GPA,
        CreditsEarned: student.CreditsEarned,
        IntendedGraduationTerm: student.IntendedGraduationTerm,
      };
      students.push(newStudent);
    }
  } catch (e) {
    console.log(`ERROR: ${e}`);
    return students;
  } finally {
    if (db) {
      await db.close();
    }
  }
  return students;
}
